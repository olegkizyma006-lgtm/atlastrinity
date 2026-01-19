import AppKit
import CoreGraphics
import CoreServices
import EventKit
import Foundation
import MCP
import MacosUseSDK
import SwiftSoup
import UserNotifications
import Vision

// --- Persistent State ---
var persistentCWD: String = FileManager.default.currentDirectoryPath

// Helper for flexible ISO8601 parsing
func parseISO8601(from string: String) -> Date? {
    let formatters: [ISO8601DateFormatter] = [
        {
            let f = ISO8601DateFormatter()
            f.formatOptions = [.withInternetDateTime]
            return f
        }(),
        {
            let f = ISO8601DateFormatter()
            f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            return f
        }(),
    ]
    for formatter in formatters {
        if let date = formatter.date(from: string) {
            return date
        }
    }
    return nil
}

// --- Vision / Screenshot Helpers ---

struct VisionElement: Encodable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct WindowActionResult: Codable {
    let action: String
    let pid: Int
    let actualX: Double
    let actualY: Double
    let actualWidth: Double
    let actualHeight: Double
    let note: String
}

func captureMainDisplay() -> CGImage? {
    return CGDisplayCreateImage(CGMainDisplayID())
}

func encodeBase64JPEG(image: CGImage) -> String? {
    let bitmapRep = NSBitmapImageRep(cgImage: image)
    guard let data = bitmapRep.representation(using: .jpeg, properties: [.compressionFactor: 0.8])
    else { return nil }
    return data.base64EncodedString()
}

func performOCR(on image: CGImage) -> [VisionElement] {
    var elements: [VisionElement] = []
    let request = VNRecognizeTextRequest { (request, error) in
        guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
        let width = Double(image.width)
        let height = Double(image.height)

        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }

            // Convert normalized Vision coordinates (bottom-left origin) to screen coordinates (top-left origin)
            // Vision: (0,0) is bottom-left, (1,1) is top-right.
            // Screen: (0,0) is top-left, (width,height) is bottom-right.

            let boundingBox = observation.boundingBox

            // X is same direction
            let x = boundingBox.origin.x * width
            let w = boundingBox.size.width * width

            // Y is flipped. Vision Bottom = 0, Screen Top = 0.
            // Screen Y = (1 - VisionMaxY) * ScreenHeight
            // VisionMaxY = origin.y + size.height
            // let visionMaxY = boundingBox.origin.y + boundingBox.size.height
            // let screenY = (1.0 - visionMaxY) * height

            // Alternate calculation:
            // boundBox.origin.y is bottom edge in normalized coord.
            // boundBox.origin.y + height is top edge in normalized coord.
            // We want Top edge in screen units (which is min Y).
            // ScreenY = (1 - (y + h)) * height
            let screenY = (1.0 - (boundingBox.origin.y + boundingBox.size.height)) * height
            let h = boundingBox.size.height * height

            // Center point for clicking
            let centerX = x + (w / 2.0)
            let centerY = screenY + (h / 2.0)

            elements.append(
                VisionElement(
                    text: candidate.string,
                    confidence: candidate.confidence,
                    x: centerX,
                    y: centerY,
                    width: w,
                    height: h
                ))
        }
    }

    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true

    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try? handler.perform([request])

    return elements
}

// --- Helper for Shell execution ---
func runShellCommand(_ command: String) -> String {
    let task = Process()
    let pipe = Pipe()
    let errorPipe = Pipe()

    task.standardOutput = pipe
    task.standardError = errorPipe
    task.arguments = ["-c", command]
    task.launchPath = "/bin/zsh"
    task.currentDirectoryPath = persistentCWD

    // Set environment variables (optional, copy current env)
    task.environment = ProcessInfo.processInfo.environment

    do {
        try task.run()

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()

        task.waitUntilExit()

        let output = String(data: data, encoding: .utf8) ?? ""
        let errorOutput = String(data: errorData, encoding: .utf8) ?? ""
        return output + errorOutput
    } catch {
        return "Failed to execute command: \(error)"
    }
}

// --- Helper to resolve PID (handles 0 or -1 for frontmost app) ---
func resolvePid(_ pid: Int?) -> Int {
    if let p = pid, p > 0 {
        return p
    }
    // Default to frontmost application
    if let frontmost = NSWorkspace.shared.frontmostApplication {
        fputs(
            "log: resolvePid: using frontmost application '\(frontmost.localizedName ?? "unknown")' (PID: \(frontmost.processIdentifier))\n",
            stderr)
        return Int(frontmost.processIdentifier)
    }
    return 0
}

// --- Persistent EventStore ---
let eventStore = EKEventStore()

// --- Spotlight Helper ---
class SpotlightSearcher: NSObject {
    var query: NSMetadataQuery?
    var semaphore: DispatchSemaphore?
    var results: [String] = []

    func search(queryStr: String) -> [String] {
        self.results = []
        self.semaphore = DispatchSemaphore(value: 0)
        self.query = NSMetadataQuery()

        guard let query = self.query else { return [] }

        query.searchScopes = [NSMetadataQueryLocalComputerScope]
        query.predicate = NSPredicate(
            format: "%K == 1 || %K LIKE[cd] %@", NSMetadataItemFSNameKey, NSMetadataItemFSNameKey,
            "*\(queryStr)*")
        // Simple filename match. Advanced usage could allow raw NSPredicate strings.

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(queryDidFinish(_:)),
            name: .NSMetadataQueryDidFinishGathering,
            object: query
        )

        query.start()

        // Timeout after 5 seconds to prevent hanging
        _ = self.semaphore?.wait(timeout: .now() + 5)

        query.stop()
        NotificationCenter.default.removeObserver(self)

        return self.results
    }

    @objc func queryDidFinish(_ notification: Foundation.Notification) {
        guard let query = notification.object as? NSMetadataQuery else { return }
        query.disableUpdates()

        for i in 0..<query.resultCount {
            if let item = query.result(at: i) as? NSMetadataItem,
                let path = item.value(forAttribute: NSMetadataItemPathKey) as? String
            {
                self.results.append(path)
            }
        }
        self.semaphore?.signal()
    }
}
let spotlight = SpotlightSearcher()

// --- Helper for AppleScript Execution ---
func runAppleScript(_ script: String) -> (success: Bool, output: String, error: String?) {
    var errorDict: NSDictionary?
    if let appleScript = NSAppleScript(source: script) {
        let descriptor = appleScript.executeAndReturnError(&errorDict)
        if let error = errorDict {
            let errorMessage =
                error[NSAppleScript.errorMessage] as? String ?? "Unknown AppleScript error"
            return (false, "", errorMessage)
        }
        return (true, descriptor.stringValue ?? "", nil)
    }
    return (false, "", "Failed to initialize NSAppleScript")
}

// --- Helper to serialize Swift structs to JSON String ---
func serializeToJsonString<T: Encodable>(_ value: T) -> String? {
    let encoder = JSONEncoder()
    // Use pretty printing for easier debugging of the output if needed
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    do {
        let jsonData = try encoder.encode(value)
        return String(data: jsonData, encoding: .utf8)
    } catch {
        fputs("error: serializeToJsonString: failed to encode value to JSON: \(error)\n", stderr)
        return nil
    }
}

// --- Function to get arguments from MCP Value ---
// Helper to extract typed values safely
func getRequiredString(from args: [String: Value]?, key: String) throws -> String {
    guard let val = args?[key]?.stringValue else {
        throw MCPError.invalidParams("Missing or invalid required string argument: '\(key)'")
    }
    return val
}

func getRequiredDouble(from args: [String: Value]?, key: String) throws -> Double {
    guard let value = args?[key] else {
        throw MCPError.invalidParams("Missing required number argument: '\(key)'")
    }
    switch value {
    case .int(let intValue):
        fputs(
            "log: getRequiredDouble: converting int \(intValue) to double for key '\(key)'\n",
            stderr)
        return Double(intValue)
    case .double(let doubleValue):
        return doubleValue
    default:
        throw MCPError.invalidParams(
            "Invalid type for required number argument: '\(key)', expected Int or Double, got \(value)"
        )
    }
}

func getRequiredInt(from args: [String: Value]?, key: String) throws -> Int {
    guard let value = args?[key] else {
        throw MCPError.invalidParams("Missing required integer argument: '\(key)'")
    }
    // Allow conversion from Double if it's an exact integer
    if let doubleValue = value.doubleValue {
        if let intValue = Int(exactly: doubleValue) {
            fputs(
                "log: getRequiredInt: converting exact double \(doubleValue) to int for key '\(key)'\n",
                stderr)
            return intValue
        } else {
            fputs(
                "warning: getRequiredInt: received non-exact double \(doubleValue) for key '\(key)', expecting integer.\n",
                stderr)
            throw MCPError.invalidParams(
                "Invalid type for required integer argument: '\(key)', received non-exact Double \(doubleValue)"
            )
        }
    }
    // Otherwise, require it to be an Int directly
    guard let intValue = value.intValue else {
        throw MCPError.invalidParams(
            "Invalid type for required integer argument: '\(key)', expected Int or exact Double, got \(value)"
        )
    }
    return intValue
}

// --- Get Optional arguments ---
// Helper for optional values
func getOptionalDouble(from args: [String: Value]?, key: String) throws -> Double? {
    guard let value = args?[key] else { return nil }  // Key not present is valid for optional
    if value.isNull { return nil }  // Explicit null is also valid
    switch value {
    case .int(let intValue):
        fputs(
            "log: getOptionalDouble: converting int \(intValue) to double for key '\(key)'\n",
            stderr)
        return Double(intValue)
    case .double(let doubleValue):
        return doubleValue
    default:
        throw MCPError.invalidParams(
            "Invalid type for optional number argument: '\(key)', expected Int or Double, got \(value)"
        )
    }
}

func getOptionalInt(from args: [String: Value]?, key: String) throws -> Int? {
    guard let value = args?[key] else { return nil }  // Key not present is valid for optional
    if value.isNull { return nil }  // Explicit null is also valid

    if let doubleValue = value.doubleValue {
        if let intValue = Int(exactly: doubleValue) {
            fputs(
                "log: getOptionalInt: converting exact double \(doubleValue) to int for key '\(key)'\n",
                stderr)
            return intValue
        } else {
            fputs(
                "warning: getOptionalInt: received non-exact double \(doubleValue) for key '\(key)', expecting integer.\n",
                stderr)
            throw MCPError.invalidParams(
                "Invalid type for optional integer argument: '\(key)', received non-exact Double \(doubleValue)"
            )
        }
    }
    guard let intValue = value.intValue else {
        throw MCPError.invalidParams(
            "Invalid type for optional integer argument: '\(key)', expected Int or exact Double, got \(value)"
        )
    }
    return intValue
}

func getOptionalBool(from args: [String: Value]?, key: String) throws -> Bool? {
    guard let value = args?[key] else { return nil }  // Key not present
    if value.isNull { return nil }  // Explicit null
    guard let boolValue = value.boolValue else {
        throw MCPError.invalidParams(
            "Invalid type for optional boolean argument: '\(key)', expected Bool, got \(value)")
    }
    return boolValue
}

func getOptionalString(from args: [String: Value]?, key: String) throws -> String? {
    guard let value = args?[key] else { return nil }
    if value.isNull { return nil }
    guard let strValue = value.stringValue else {
        throw MCPError.invalidParams(
            "Invalid type for optional string argument: '\(key)', expected String, got \(value)")
    }
    return strValue
}

// --- NEW Helper to parse modifier flags ---
func parseFlags(from value: Value?) throws -> CGEventFlags {
    guard let arrayValue = value?.arrayValue else {
        // No flags provided or not an array, return empty flags
        return []
    }

    var flags: CGEventFlags = []
    for flagValue in arrayValue {
        guard let flagString = flagValue.stringValue else {
            throw MCPError.invalidParams(
                "Invalid modifierFlags array: contains non-string element \(flagValue)")
        }
        switch flagString.lowercased() {
        // Standard modifiers
        case "capslock", "caps": flags.insert(.maskAlphaShift)
        case "shift": flags.insert(.maskShift)
        case "control", "ctrl": flags.insert(.maskControl)
        case "option", "opt", "alt": flags.insert(.maskAlternate)
        case "command", "cmd": flags.insert(.maskCommand)
        // Other potentially useful flags
        case "help": flags.insert(.maskHelp)
        case "function", "fn": flags.insert(.maskSecondaryFn)
        case "numericpad", "numpad": flags.insert(.maskNumericPad)
        // Non-keyed state (less common for press simulation)
        // case "noncoalesced": flags.insert(.maskNonCoalesced)
        default:
            fputs(
                "warning: parseFlags: unknown modifier flag string '\(flagString)', ignoring.\n",
                stderr)
        // Optionally throw an error:
        // throw MCPError.invalidParams("Unknown modifier flag: '\(flagString)'")
        }
    }
    return flags
}

// Async helper function to set up and start the server
func setupAndStartServer() async throws -> Server {
    fputs("log: setupAndStartServer: entering function.\n", stderr)

    // --- Define Schemas and Tools for Simplified Actions ---
    // (Schemas remain the same as they define the MCP interface)
    let openAppSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "identifier": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. App name, path, or bundle ID."),
            ])
        ]),
        "required": .array([.string("identifier")]),
    ])
    let openAppTool = Tool(
        name: "macos-use_open_application_and_traverse",
        description: "Opens/activates an application and then traverses its accessibility tree.",
        inputSchema: openAppSchema
    )

    let clickSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application. Defaults to frontmost app."),
            ]),
            "x": .object([
                "type": .string("number"),
                "description": .string("REQUIRED. X coordinate for the click."),
            ]),
            "y": .object([
                "type": .string("number"),
                "description": .string("REQUIRED. Y coordinate for the click."),
            ]),
            "showAnimation": .object([
                "type": .string("boolean"),
                "description": .string("OPTIONAL. Show visual feedback animation (green circle)."),
            ]),
            "animationDuration": .object([
                "type": .string("number"),
                "description": .string("OPTIONAL. Duration of the animation in seconds."),
            ]),
        ]),
        "required": .array([.string("x"), .string("y")]),
    ])
    let clickTool = Tool(
        name: "macos-use_click_and_traverse",
        description:
            "Simulates a click at the given coordinates within the app specified by PID, then traverses its accessibility tree.",
        inputSchema: clickSchema
    )

    let typeSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application window. Defaults to frontmost app."),
            ]),
            "text": .object([
                "type": .string("string"), "description": .string("REQUIRED. Text to type."),
            ]),
            // Add optional options here if needed later
        ]),
        "required": .array([.string("text")]),
    ])
    let typeTool = Tool(
        name: "macos-use_type_and_traverse",
        description:
            "Simulates typing text into the app specified by PID, then traverses its accessibility tree.",
        inputSchema: typeSchema
    )

    let refreshSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the application to traverse. Defaults to frontmost app."),
            ])
            // Add optional options here if needed later
        ]),
        "required": .array([]),
    ])
    let refreshTool = Tool(
        name: "macos-use_refresh_traversal",
        description: "Traverses the accessibility tree of the application specified by PID.",
        inputSchema: refreshSchema
    )

    // *** NEW: Schema and Tool for Execute Command ***
    let executeCommandSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "command": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. The shell command to execute."),
            ])
        ]),
        "required": .array([.string("command")]),
    ])
    let executeCommandTool = Tool(
        name: "execute_command",  // Matching the Python terminal MCP name
        description: "Execute a terminal command in a persistent shell session (maintains CWD).",
        inputSchema: executeCommandSchema
    )
    let terminalTool = Tool(
        name: "terminal",
        description: "Alias for execute_command.",
        inputSchema: executeCommandSchema
    )

    // *** NEW: Screenshot Tool ***
    let screenshotSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([:]),
    ])
    let screenshotTool = Tool(
        name: "macos-use_take_screenshot",
        description:
            "Captures a screenshot of the main display and returns it as a Base64 encoded PNG string.",
        inputSchema: screenshotSchema
    )
    let screenshotAliasTool = Tool(
        name: "screenshot",
        description: "Alias for macos-use_take_screenshot.",
        inputSchema: screenshotSchema
    )

    // *** NEW: Vision Analysis Tool ***
    let visionSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([:]),
    ])
    let visionTool = Tool(
        name: "macos-use_analyze_screen",
        description:
            "Uses native Apple Vision framework to detect text (OCR) on the screen. Returns coordinates and text content.",
        inputSchema: visionSchema
    )
    let ocrAliasTool = Tool(
        name: "ocr",
        description: "Alias for macos-use_analyze_screen.",
        inputSchema: visionSchema
    )
    let analyzeAliasTool = Tool(
        name: "analyze",
        description: "Alias for macos-use_analyze_screen.",
        inputSchema: visionSchema
    )

    // *** NEW: Schema and Tool for Press Key ***
    let pressKeySchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application window. Defaults to frontmost app."),
            ]),
            "keyName": .object([
                "type": .string("string"),
                "description": .string(
                    "REQUIRED. Name of the key to press (e.g., 'Return', 'Enter', 'Escape', 'Tab', 'ArrowUp', 'Delete', 'a', 'B'). Case-sensitive for letter keys if no modifiers used."
                ),
            ]),
            "modifierFlags": .object([  // Optional array of strings
                "type": .string("array"),
                "description": .string(
                    "OPTIONAL. Modifier keys to hold (e.g., ['Command', 'Shift']). Valid: CapsLock, Shift, Control, Option, Command, Function, NumericPad, Help."
                ),
                "items": .object(["type": .string("string")]),  // Items in the array must be strings
            ]),
            // Add optional ActionOptions overrides here if needed later
        ]),
        "required": .array([.string("keyName")]),
    ])
    let pressKeyTool = Tool(
        name: "macos-use_press_key_and_traverse",
        description:
            "Simulates pressing a specific key (like Return, Enter, Escape, Tab, Arrow Keys, regular characters) with optional modifiers, then traverses the accessibility tree.",
        inputSchema: pressKeySchema
    )

    // *** NEW: Scroll Tool ***
    let scrollSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application. Defaults to frontmost app."),
            ]),
            "direction": .object([
                "type": .string("string"),
                "description": .string(
                    "REQUIRED. Direction to scroll: 'up', 'down', 'left', 'right'."),
            ]),
            "amount": .object([
                "type": .string("number"),
                "description": .string("OPTIONAL. Amount to scroll (default 3)."),
            ]),
        ]),
        "required": .array([.string("direction")]),
    ])
    let scrollTool = Tool(
        name: "macos-use_scroll_and_traverse",
        description: "Simulates a mouse scroll wheel action in a specific direction.",
        inputSchema: scrollSchema
    )

    // *** NEW: Right Click Tool ***
    let mouseActionSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application. Defaults to frontmost app."),
            ]),
            "x": .object([
                "type": .string("number"), "description": .string("REQUIRED. Screen X coordinate."),
            ]),
            "y": .object([
                "type": .string("number"), "description": .string("REQUIRED. Screen Y coordinate."),
            ]),
        ]),
        "required": .array([.string("x"), .string("y")]),
    ])
    let rightClickTool = Tool(
        name: "macos-use_right_click_and_traverse",
        description: "Simulates a right-click (context menu) at the specified coordinates.",
        inputSchema: mouseActionSchema
    )
    let doubleClickTool = Tool(
        name: "macos-use_double_click_and_traverse",
        description: "Simulates a double-click at the specified coordinates.",
        inputSchema: mouseActionSchema
    )

    // *** NEW: Drag & Drop Tool ***
    let dragDropSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the target application. Defaults to frontmost app."),
            ]),
            "startX": .object([
                "type": .string("number"), "description": .string("REQUIRED. Start X coordinate."),
            ]),
            "startY": .object([
                "type": .string("number"), "description": .string("REQUIRED. Start Y coordinate."),
            ]),
            "endX": .object([
                "type": .string("number"), "description": .string("REQUIRED. End X coordinate."),
            ]),
            "endY": .object([
                "type": .string("number"), "description": .string("REQUIRED. End Y coordinate."),
            ]),
        ]),
        "required": .array([
            .string("startX"), .string("startY"), .string("endX"), .string("endY"),
        ]),
    ])
    let dragDropTool = Tool(
        name: "macos-use_drag_and_drop_and_traverse",
        description: "Simulates a mouse drag-and-drop action.",
        inputSchema: dragDropSchema
    )

    let windowMgmtSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "pid": .object([
                "type": .string("number"),
                "description": .string(
                    "OPTIONAL. PID of the application. Defaults to frontmost app."),
            ]),
            "action": .object([
                "type": .string("string"),
                "description": .string(
                    "REQUIRED. Action: 'move', 'resize', 'minimize', 'maximize', 'make_front'."),
            ]),
            "x": .object([
                "type": .string("number"), "description": .string("Optional X for move."),
            ]),
            "y": .object([
                "type": .string("number"), "description": .string("Optional Y for move."),
            ]),
            "width": .object([
                "type": .string("number"), "description": .string("Optional Width for resize."),
            ]),
            "height": .object([
                "type": .string("number"), "description": .string("Optional Height for resize."),
            ]),
        ]),
        "required": .array([.string("action")]),
    ])
    let windowMgmtTool = Tool(
        name: "macos-use_window_management",
        description: "Manages application windows (move, resize, minimize, maximize, make_front).",
        inputSchema: windowMgmtSchema
    )

    // *** NEW: Clipboard Tools ***
    let setClipboardSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "text": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. Text to set to clipboard."),
            ]),
            "showAnimation": .object([
                "type": .string("boolean"),
                "description": .string("OPTIONAL. Show visual focus animation."),
            ]),
            "animationDuration": .object([
                "type": .string("number"),
                "description": .string("OPTIONAL. Duration of the animation in seconds."),
            ]),
        ]),
        "required": .array([.string("text")]),
    ])
    let setClipboardTool = Tool(
        name: "macos-use_set_clipboard",
        description: "Sets the system clipboard text content.",
        inputSchema: setClipboardSchema
    )
    let getClipboardTool = Tool(
        name: "macos-use_get_clipboard",
        description: "Returns the current system clipboard text content.",
        inputSchema: .object(["type": .string("object"), "properties": .object([:])])
    )

    // *** NEW: Media Control Tool ***
    let mediaControlSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "action": .object([
                "type": .string("string"),
                "description": .string(
                    "Action: 'play_pause', 'next', 'previous', 'volume_up', 'volume_down', 'mute', 'brightness_up', 'brightness_down'."
                ),
            ])
        ]),
        "required": .array([.string("action")]),
    ])
    let mediaControlTool = Tool(
        name: "macos-use_system_control",
        description: "Controls system media, volume, and brightness.",
        inputSchema: mediaControlSchema
    )

    // *** NEW: Fetch URL Tool ***
    let fetchRelaySchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "url": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. The URL to fetch."),
            ])
        ]),
        "required": .array([.string("url")]),
    ])
    let fetchTool = Tool(
        name: "macos-use_fetch_url",
        description:
            "Fetches content from a URL and converts HTML to text/markdown using SwiftSoup.",
        inputSchema: fetchRelaySchema
    )

    // *** NEW: Get Time Tool ***
    let getTimeSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "timezone": .object([
                "type": .string("string"),
                "description": .string(
                    "OPTIONAL. Timezone identifier (e.g., 'America/Los_Angeles'). Defaults to system."
                ),
            ])
        ]),
    ])
    let getTimeTool = Tool(
        name: "macos-use_get_time",
        description: "Returns the current system time.",
        inputSchema: getTimeSchema
    )

    // *** NEW: AppleScript Tool ***
    let appleScriptSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "script": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. The AppleScript code to execute."),
            ])
        ]),
        "required": .array([.string("script")]),
    ])
    let appleScriptTool = Tool(
        name: "macos-use_run_applescript",
        description: "Executes an arbitrary AppleScript.",
        inputSchema: appleScriptSchema
    )

    // *** NEW: Calendar Tools ***
    let calendarEventsSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "start": .object([
                "type": .string("string"),
                "description": .string("Start date (ISO8601 or natural language)"),
            ]),
            "end": .object([
                "type": .string("string"),
                "description": .string("End date (ISO8601 or natural language)"),
            ]),
        ]),
        "required": .array([.string("start"), .string("end")]),
    ])
    let calendarEventsTool = Tool(
        name: "macos-use_calendar_events",
        description: "Fetch calendar events for a date range.",
        inputSchema: calendarEventsSchema
    )

    let createEventSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "title": .object(["type": .string("string"), "description": .string("Event title")]),
            "date": .object(["type": .string("string"), "description": .string("Event date")]),
        ]),
        "required": .array([.string("title"), .string("date")]),
    ])
    let createEventTool = Tool(
        name: "macos-use_create_event",
        description: "Create a calendar event.",
        inputSchema: createEventSchema
    )

    // *** NEW: Reminder Tools ***
    let remindersSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "list": .object([
                "type": .string("string"), "description": .string("Optional list name filter"),
            ])
        ]),
    ])
    let remindersTool = Tool(
        name: "macos-use_reminders",
        description: "Fetch incomplete reminders.",
        inputSchema: remindersSchema
    )

    let createReminderSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "title": .object(["type": .string("string"), "description": .string("Reminder title")])
        ]),
        "required": .array([.string("title")]),
    ])
    let createReminderTool = Tool(
        name: "macos-use_create_reminder",
        description: "Create a new reminder.",
        inputSchema: createReminderSchema
    )

    // *** NEW: Spotlight Tool ***
    let spotlightSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "query": .object([
                "type": .string("string"), "description": .string("Filename search query"),
            ])
        ]),
        "required": .array([.string("query")]),
    ])
    let spotlightTool = Tool(
        name: "macos-use_spotlight_search",
        description: "Search for files using Spotlight (mdfind).",
        inputSchema: spotlightSchema
    )

    // *** NEW: Notification Tool ***
    let notificationSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "title": .object([
                "type": .string("string"), "description": .string("Notification title"),
            ]),
            "message": .object([
                "type": .string("string"), "description": .string("Notification body text"),
            ]),
        ]),
        "required": .array([.string("title"), .string("message")]),
    ])
    let notificationTool = Tool(
        name: "macos-use_send_notification",
        description: "Send a native macOS system notification.",
        inputSchema: notificationSchema
    )

    // *** NEW: Apple Notes Tools ***
    let notesListFoldersSchema: Value = .object([
        "type": .string("object"), "properties": .object([:]),
    ])
    let notesListFoldersTool = Tool(
        name: "macos-use_notes_list_folders",
        description: "List all folders in Apple Notes.",
        inputSchema: notesListFoldersSchema
    )

    let notesCreateSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "body": .object([
                "type": .string("string"),
                "description": .string("HTML or plain text content. First line becomes title."),
            ]),
            "folder": .object([
                "type": .string("string"), "description": .string("Optional folder name."),
            ]),
        ]),
        "required": .array([.string("body")]),
    ])
    let notesCreateTool = Tool(
        name: "macos-use_notes_create_note",
        description: "Create a new note in Apple Notes.",
        inputSchema: notesCreateSchema
    )

    let notesGetSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "name": .object([
                "type": .string("string"),
                "description": .string("Name/Title of the note to find."),
            ])
        ]),
        "required": .array([.string("name")]),
    ])
    let notesGetTool = Tool(
        name: "macos-use_notes_get_content",
        description: "Get the HTML content of a note by name.",
        inputSchema: notesGetSchema
    )

    // *** NEW: Apple Mail Tools ***
    let mailSendSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "to": .object([
                "type": .string("string"), "description": .string("Recipient email address"),
            ]),
            "subject": .object(["type": .string("string"), "description": .string("Subject line")]),
            "body": .object([
                "type": .string("string"), "description": .string("Email body content"),
            ]),
        ]),
        "required": .array([.string("to"), .string("subject"), .string("body")]),
    ])
    let mailSendTool = Tool(
        name: "macos-use_mail_send",
        description: "Send an email using Apple Mail.",
        inputSchema: mailSendSchema
    )

    let mailReadSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "limit": .object([
                "type": .string("integer"),
                "description": .string("Number of recent messages to read (default 5)"),
            ])
        ]),
    ])
    let mailReadTool = Tool(
        name: "macos-use_mail_read_inbox",
        description: "Read recent subject lines from Apple Mail Inbox.",
        inputSchema: mailReadSchema
    )

    // *** NEW: Finder Tools ***
    let finderListFilesSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "path": .object([
                "type": .string("string"),
                "description": .string(
                    "Optional folder path. If omitted, uses frontmost Finder window."),
            ])
        ]),
    ])
    let finderListFilesTool = Tool(
        name: "macos-use_finder_list_files",
        description:
            "Lists files and folders in the frontmost Finder window or at a specified path.",
        inputSchema: finderListFilesSchema
    )

    let finderGetSelectionSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([:]),
    ])
    let finderGetSelectionTool = Tool(
        name: "macos-use_finder_get_selection",
        description: "Returns the POSIX paths of currently selected items in Finder.",
        inputSchema: finderGetSelectionSchema
    )

    let finderOpenPathSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "path": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. The POSIX path to open."),
            ])
        ]),
        "required": .array([.string("path")]),
    ])
    let finderOpenPathTool = Tool(
        name: "macos-use_finder_open_path",
        description: "Opens a folder or file in Finder.",
        inputSchema: finderOpenPathSchema
    )

    let finderMoveToTrashSchema: Value = .object([
        "type": .string("object"),
        "properties": .object([
            "path": .object([
                "type": .string("string"),
                "description": .string("REQUIRED. The POSIX path of the item to trash."),
            ])
        ]),
        "required": .array([.string("path")]),
    ])
    let finderMoveToTrashTool = Tool(
        name: "macos-use_finder_move_to_trash",
        description: "Moves the specified file or folder to the Trash via Finder.",
        inputSchema: finderMoveToTrashSchema
    )

    // *** NEW: Dynamic Help ***
    let dynamicHelpSchema: Value = .object([
        "type": .string("object"), "properties": .object([:]),
    ])
    let dynamicHelpTool = Tool(
        name: "macos-use_list_tools_dynamic",
        description:
            "Returns a detailed JSON structure describing all available tools, their schemas, and usage examples.",
        inputSchema: dynamicHelpSchema
    )

    // --- Aggregate list of tools ---
    let allTools = [
        openAppTool, clickTool, rightClickTool, doubleClickTool, dragDropTool, typeTool,
        pressKeyTool,
        scrollTool, refreshTool, windowMgmtTool, executeCommandTool, terminalTool,
        screenshotTool, screenshotAliasTool, visionTool, ocrAliasTool, analyzeAliasTool,
        setClipboardTool, getClipboardTool, mediaControlTool, fetchTool, getTimeTool,
        appleScriptTool, calendarEventsTool, createEventTool, remindersTool, createReminderTool,
        spotlightTool, notificationTool, notesListFoldersTool, notesCreateTool, notesGetTool,
        mailSendTool, mailReadTool,
        finderListFilesTool, finderGetSelectionTool, finderOpenPathTool, finderMoveToTrashTool,
        dynamicHelpTool,
    ]
    fputs(
        "log: setupAndStartServer: defined \(allTools.count) tools: \(allTools.map { $0.name })\n",
        stderr)

    let server = Server(
        name: "SwiftMacOSServerDirect",  // Renamed slightly
        version: "1.3.0",  // Incremented version for major change
        capabilities: .init(
            tools: .init(listChanged: true)
        )
    )
    fputs(
        "log: setupAndStartServer: server instance created (\(server.name)) version \(server.version).\n",
        stderr)

    // --- Dummy Handlers (ReadResource, ListResources, ListPrompts) ---
    // (Keep these as they are part of the MCP spec, even if unused for now)
    await server.withMethodHandler(ReadResource.self) { params in
        let uri = params.uri
        fputs(
            "log: handler(ReadResource): received request for uri: \(uri) (dummy handler)\n", stderr
        )
        // In a real scenario, you might fetch resource content here
        return .init(contents: [.text("dummy content for \(uri)", uri: uri)])
    }
    fputs("log: setupAndStartServer: registered ReadResource handler (dummy).\n", stderr)

    await server.withMethodHandler(ListResources.self) { _ in
        fputs("log: handler(ListResources): received request (dummy handler).\n", stderr)
        // In a real scenario, list available resources
        return ListResources.Result(resources: [])
    }
    fputs("log: setupAndStartServer: registered ListResources handler (dummy).\n", stderr)

    await server.withMethodHandler(ListPrompts.self) { _ in
        fputs("log: handler(ListPrompts): received request (dummy handler).\n", stderr)
        // In a real scenario, list available prompts
        return ListPrompts.Result(prompts: [])
    }
    fputs("log: setupAndStartServer: registered ListPrompts handler (dummy).\n", stderr)

    // --- ListTools Handler ---
    await server.withMethodHandler(ListTools.self) { _ in
        fputs("log: handler(ListTools): received request.\n", stderr)
        let result = ListTools.Result(tools: allTools)
        fputs(
            "log: handler(ListTools): responding with \(result.tools.count) tools: \(result.tools.map { $0.name })\n",
            stderr)
        return result
    }
    fputs("log: setupAndStartServer: registered ListTools handler.\n", stderr)

    // --- UPDATED CallTool Handler (Direct SDK Call) ---
    await server.withMethodHandler(CallTool.self) { params in
        fputs("log: handler(CallTool): received request for tool: \(params.name).\n", stderr)
        fputs(
            "log: handler(CallTool): arguments received (raw MCP): \(params.arguments?.debugDescription ?? "nil")\n",
            stderr)

        do {
            // --- Determine Action and Options from MCP Params ---
            let primaryAction: PrimaryAction
            var options = ActionOptions()  // Start with default options
            options.showAnimation = true  // ENABLE ANIMATION BY DEFAULT
            options.animationDuration = 0.8  // 0.8s for good visibility

            // PID is optional (defaults to frontmost app if 0, -1 or missing)
            let pidOptionalInt = try getOptionalInt(from: params.arguments, key: "pid")
            let resolvedPid = resolvePid(pidOptionalInt)

            // Convert to pid_t
            guard let convertedPid = pid_t(exactly: resolvedPid) else {
                fputs(
                    "error: handler(CallTool): Resolved PID value \(resolvedPid) is out of range for pid_t.\n",
                    stderr)
                throw MCPError.invalidParams("Resolved PID value \(resolvedPid) is out of range.")
            }
            options.pidForTraversal = convertedPid

            // Potentially allow overriding default options from params
            options.traverseBefore =
                try getOptionalBool(from: params.arguments, key: "traverseBefore")
                ?? options.traverseBefore
            options.traverseAfter =
                try getOptionalBool(from: params.arguments, key: "traverseAfter")
                ?? options.traverseAfter
            options.showDiff =
                try getOptionalBool(from: params.arguments, key: "showDiff") ?? options.showDiff
            options.onlyVisibleElements =
                try getOptionalBool(from: params.arguments, key: "onlyVisibleElements")
                ?? options.onlyVisibleElements
            options.showAnimation =
                try getOptionalBool(from: params.arguments, key: "showAnimation")
                ?? options.showAnimation
            options.animationDuration =
                try getOptionalDouble(from: params.arguments, key: "animationDuration")
                ?? options.animationDuration
            options.delayAfterAction =
                try getOptionalDouble(from: params.arguments, key: "delayAfterAction")
                ?? options.delayAfterAction

            options = options.validated()
            fputs("log: handler(CallTool): constructed ActionOptions: \(options)\n", stderr)

            switch params.name {
            case openAppTool.name:
                let identifier = try getRequiredString(from: params.arguments, key: "identifier")
                primaryAction = .open(identifier: identifier)

            case clickTool.name:
                let x = try getRequiredDouble(from: params.arguments, key: "x")
                let y = try getRequiredDouble(from: params.arguments, key: "y")
                primaryAction = .input(action: .click(point: CGPoint(x: x, y: y)))
                options.pidForTraversal = convertedPid  // Re-affirm

            case typeTool.name:
                let text = try getRequiredString(from: params.arguments, key: "text")
                primaryAction = .input(action: .type(text: text))
                options.pidForTraversal = convertedPid  // Re-affirm

            // ... (Other existing cases) ...

            case fetchTool.name:
                let urlString = try getRequiredString(from: params.arguments, key: "url")
                guard let url = URL(string: urlString) else {
                    return CallTool.Result(
                        content: [.text("Invalid URL: \(urlString)")], isError: true)
                }

                let (resultText, isError): (String, Bool) = await withCheckedContinuation { continuation in
                    let task = URLSession.shared.dataTask(with: url) { data, response, error in
                        var text = ""
                        var errorFound = false
                        if let error = error {
                            text = "Error fetching URL: \(error.localizedDescription)"
                            errorFound = true
                        } else if let data = data, let html = String(data: data, encoding: .utf8) {
                            do {
                                let doc = try SwiftSoup.parse(html)
                                text = try doc.text()
                            } catch {
                                text = "Error parsing HTML: \(error.localizedDescription)"
                                errorFound = true
                            }
                        } else {
                            text = "No data or invalid encoding."
                            errorFound = true
                        }
                        continuation.resume(returning: (text, errorFound))
                    }
                    task.resume()
                }
                return CallTool.Result(content: [.text(resultText)], isError: isError)

            case getTimeTool.name:
                let timezone = try getOptionalString(from: params.arguments, key: "timezone")  // Need helper
                let formatter = DateFormatter()
                formatter.dateStyle = .full
                formatter.timeStyle = .full
                if let tz = timezone {
                    if let timeZone = TimeZone(identifier: tz) {
                        formatter.timeZone = timeZone
                    } else {
                        return CallTool.Result(
                            content: [.text("Invalid timezone identifier: \(tz)")], isError: true)
                    }
                }
                return CallTool.Result(content: [.text(formatter.string(from: Date()))])

            case appleScriptTool.name:
                let script = try getRequiredString(from: params.arguments, key: "script")
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("AppleScript Error: \(error ?? "Unknown")")], isError: true)
                }

            // --- Handlers for Universal Tools ---

            case calendarEventsTool.name:
                let startStr = try getRequiredString(from: params.arguments, key: "start")
                let endStr = try getRequiredString(from: params.arguments, key: "end")
                // Very basic ISO parsing for now, user should provide ISO8601
                let isoFormatter = ISO8601DateFormatter()
                isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

                guard let start = parseISO8601(from: startStr),
                    let end = parseISO8601(from: endStr)
                else {
                    return CallTool.Result(
                        content: [
                            .text("Invalid date format. Use ISO8601 (e.g. 2024-01-01T12:00:00Z).")
                        ], isError: true)
                }

                let resultText: String = await withCheckedContinuation { continuation in
                    eventStore.requestAccess(to: .event) { granted, error in
                        if !granted {
                            continuation.resume(returning: "Access denied")
                            return
                        }

                        let predicate = eventStore.predicateForEvents(
                            withStart: start, end: end, calendars: nil)
                        let events = eventStore.events(matching: predicate)
                        let text = events.map {
                            "\($0.title ?? "No Title") (\($0.startDate.description) - \($0.endDate.description))"
                        }.joined(separator: "\n")
                        continuation.resume(returning: text)
                    }
                }
                return CallTool.Result(content: [.text(resultText)])

            case createEventTool.name:
                let title = try getRequiredString(from: params.arguments, key: "title")
                let dateStr = try getRequiredString(from: params.arguments, key: "date")
                let isoFormatter = ISO8601DateFormatter()
                isoFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                guard let date = parseISO8601(from: dateStr) else {
                    return CallTool.Result(
                        content: [
                            .text("Invalid date format. Use ISO8601 (e.g. 2024-01-01T12:00:00Z).")
                        ], isError: true)
                }

                let resultText: String = await withCheckedContinuation { continuation in
                    eventStore.requestAccess(to: .event) { granted, error in
                        if !granted {
                            continuation.resume(returning: "Access denied")
                            return
                        }

                        let event = EKEvent(eventStore: eventStore)
                        event.title = title
                        event.startDate = date
                        event.endDate = date.addingTimeInterval(3600)  // Default 1h
                        event.calendar = eventStore.defaultCalendarForNewEvents
                        do {
                            try eventStore.save(event, span: .thisEvent)
                            continuation.resume(returning: "Event created.")
                        } catch {
                            continuation.resume(returning: "Error: \(error)")
                        }
                    }
                }
                return CallTool.Result(content: [.text(resultText)])

            case remindersTool.name:
                let resultText: String = await withCheckedContinuation { continuation in
                    eventStore.requestAccess(to: .reminder) { granted, error in
                        if !granted {
                            continuation.resume(returning: "Access denied")
                            return
                        }
                        let predicate = eventStore.predicateForIncompleteReminders(
                            withDueDateStarting: nil, ending: nil, calendars: nil)
                        eventStore.fetchReminders(matching: predicate) { reminders in
                            let text = reminders?.map { $0.title ?? "No Title" }.joined(separator: "\n")
                                ?? "No reminders"
                            continuation.resume(returning: text)
                        }
                    }
                }
                return CallTool.Result(content: [.text(resultText)])

            case createReminderTool.name:
                let title = try getRequiredString(from: params.arguments, key: "title")
                let resultText: String = await withCheckedContinuation { continuation in
                    eventStore.requestAccess(to: .reminder) { granted, error in
                        if !granted {
                            continuation.resume(returning: "Access denied")
                            return
                        }
                        let reminder = EKReminder(eventStore: eventStore)
                        reminder.title = title
                        reminder.calendar = eventStore.defaultCalendarForNewReminders()
                        do {
                            try eventStore.save(reminder, commit: true)
                            continuation.resume(returning: "Reminder saved.")
                        } catch {
                            continuation.resume(returning: "Error: \(error)")
                        }
                    }
                }
                return CallTool.Result(content: [.text(resultText)])

            case spotlightTool.name:
                let query = try getRequiredString(from: params.arguments, key: "query")
                let results = spotlight.search(queryStr: query)
                return CallTool.Result(content: [.text(results.joined(separator: "\n"))])

            case notificationTool.name:
                let title = try getRequiredString(from: params.arguments, key: "title")
                let message = try getRequiredString(from: params.arguments, key: "message")

                let script =
                    "display notification \"\(message.replacingOccurrences(of: "\"", with: "\\\""))\" with title \"\(title.replacingOccurrences(of: "\"", with: "\\\""))\""
                _ = runShellCommand("osascript -e '\(script)'")
                return CallTool.Result(content: [.text("Notification sent via AppleScript.")])

            // --- Notes Handlers ---
            case notesListFoldersTool.name:
                let script = """
                    tell application "Notes"
                        set folderNames to name of every folder
                        return folderNames
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("Error: \(error ?? "Unknown")")], isError: true)
                }

            case notesCreateTool.name:
                let body = try getRequiredString(from: params.arguments, key: "body")
                let folder = try getOptionalString(from: params.arguments, key: "folder") ?? "Notes"
                // Escape quotes for AppleScript
                let safeBody = body.replacingOccurrences(of: "\"", with: "\\\"")
                let safeFolder = folder.replacingOccurrences(of: "\"", with: "\\\"")

                let script = """
                    tell application "Notes"
                        if not (exists folder "\(safeFolder)") then
                            return "Error: Folder '\(safeFolder)' not found."
                        end if
                        make new note at folder "\(safeFolder)" with properties {body:"\(safeBody)"}
                        return "Note created."
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("Error: \(error ?? "Unknown")")], isError: true)
                }

            case notesGetTool.name:
                let name = try getRequiredString(from: params.arguments, key: "name")
                let safeName = name.replacingOccurrences(of: "\"", with: "\\\"")
                let script = """
                    tell application "Notes"
                        try
                            set theNote to item 1 of (every note whose name contains "\(safeName)")
                            return body of theNote
                        on error
                            return "Note not found."
                        end try
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("Error: \(error ?? "Unknown")")], isError: true)
                }

            // --- Mail Handlers ---
            case mailSendTool.name:
                let to = try getRequiredString(from: params.arguments, key: "to")
                let subject = try getRequiredString(from: params.arguments, key: "subject")
                let body = try getRequiredString(from: params.arguments, key: "body")

                let safeTo = to.replacingOccurrences(of: "\"", with: "\\\"")
                let safeSubject = subject.replacingOccurrences(of: "\"", with: "\\\"")
                let safeBody = body.replacingOccurrences(of: "\"", with: "\\\"")

                let script = """
                    tell application "Mail"
                        set newMessage to make new outgoing message with properties {subject:"\(safeSubject)", content:"\(safeBody)", visible:true}
                        tell newMessage
                            make new to recipient at end of to recipients with properties {address:"\(safeTo)"}
                            send
                        end tell
                        return "Email sent."
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("Error: \(error ?? "Unknown")")], isError: true)
                }

            case mailReadTool.name:
                let limit = try getOptionalInt(from: params.arguments, key: "limit") ?? 5
                let script = """
                    tell application "Mail"
                        set inboxMessages to messages of inbox
                        set msgCount to count of inboxMessages
                        if msgCount is 0 then return "Inbox is empty"
                        
                        set resultList to {}
                        set loopLimit to \(limit)
                        if msgCount < loopLimit then set loopLimit to msgCount
                        
                        repeat with i from 1 to loopLimit
                            set msg to item i of inboxMessages
                            set end of resultList to (subject of msg) & " [From: " & (sender of msg) & "]"
                        end repeat
                        return resultList as string
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                if success {
                    return CallTool.Result(content: [.text(output)])
                } else {
                    return CallTool.Result(
                        content: [.text("Error: \(error ?? "Unknown")")], isError: true)
                }

            // --- Finder Handlers ---
            case finderListFilesTool.name:
                let path = try getOptionalString(from: params.arguments, key: "path")
                let script: String
                if let p = path {
                    script = """
                        tell application "Finder"
                            try
                                set targetFolder to POSIX file "\(p.replacingOccurrences(of: "\"", with: "\\\""))" as alias
                                set fileList to name of every item of targetFolder
                                set AppleScript's text item delimiters to ", "
                                return fileList as string
                            on error err
                                return "Error: " & err
                            end try
                        end tell
                        """
                } else {
                    script = """
                        tell application "Finder"
                            try
                                set targetFolder to target of front window
                                set fileList to name of every item of targetFolder
                                set AppleScript's text item delimiters to ", "
                                return fileList as string
                            on error
                                return "No Finder window open."
                            end try
                        end tell
                        """
                }
                let (success, output, error) = runAppleScript(script)
                return .init(
                    content: [.text(success ? output : "Error: \(error ?? "Unknown")")],
                    isError: !success)

            case finderGetSelectionTool.name:
                let script = """
                    tell application "Finder"
                        set theSelection to selection
                        if (count of theSelection) is 0 then return "Nothing selected."
                        set pathList to {}
                        repeat with anItem in theSelection
                            set end of pathList to POSIX path of (anItem as text)
                        end repeat
                        set AppleScript's text item delimiters to "\n"
                        return pathList as string
                    end tell
                    """
                let (success, output, error) = runAppleScript(script)
                return .init(
                    content: [.text(success ? output : "Error: \(error ?? "Unknown")")],
                    isError: !success)

            case finderOpenPathTool.name:
                let path = try getRequiredString(from: params.arguments, key: "path")
                let script =
                    "tell application \"Finder\" to open POSIX file \"\(path.replacingOccurrences(of: "\"", with: "\\\""))\""
                let (success, _, error) = runAppleScript(script)
                return .init(
                    content: [
                        .text(success ? "Opened path \(path)" : "Error: \(error ?? "Unknown")")
                    ],
                    isError: !success)

            case finderMoveToTrashTool.name:
                let path = try getRequiredString(from: params.arguments, key: "path")
                let script =
                    "tell application \"Finder\" to delete POSIX file \"\(path.replacingOccurrences(of: "\"", with: "\\\""))\""
                let (success, _, error) = runAppleScript(script)
                return .init(
                    content: [
                        .text(success ? "Moved to trash: \(path)" : "Error: \(error ?? "Unknown")")
                    ], isError: !success)

            // --- Dynamic Help Handler ---
            case dynamicHelpTool.name:
                // Manually construct JSON description of allTools
                // In a perfect world, we'd use Encodable on Tool, but Tool struct is from library
                // We will build a simple string representation
                var helpText = "Available Tools:\n"
                for tool in allTools {
                    helpText += "- \(tool.name): \(tool.description)\n"
                }
                return CallTool.Result(content: [.text(helpText)])  // Simplified for now

            // --- End Universal Handlers ---

            case pressKeyTool.name:

                let keyName = try getRequiredString(from: params.arguments, key: "keyName")
                // Parse optional flags using the new helper
                let flags = try parseFlags(from: params.arguments?["modifierFlags"])
                fputs("log: handler(CallTool): parsed modifierFlags: \(flags)\n", stderr)
                primaryAction = .input(action: .press(keyName: keyName, flags: flags))
                options.pidForTraversal = convertedPid  // Re-affirm

            case refreshTool.name:
                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid  // Re-affirm

            case executeCommandTool.name, terminalTool.name:
                let command = try getRequiredString(from: params.arguments, key: "command")

                // Handle "cd" command manually for persistence
                let parts = command.split(separator: " ").map(String.init)
                if parts.first == "cd" {
                    let targetDir: String
                    if parts.count > 1 {
                        let path = parts[1]
                        if path.hasPrefix("/") {
                            targetDir = path
                        } else if path == "~" {
                            targetDir = FileManager.default.homeDirectoryForCurrentUser.path
                        } else if path.hasPrefix("~/") {
                            targetDir =
                                FileManager.default.homeDirectoryForCurrentUser.path + "/"
                                + String(path.dropFirst(2))
                        } else {
                            targetDir = (persistentCWD as NSString).appendingPathComponent(path)
                        }
                    } else {
                        targetDir = FileManager.default.homeDirectoryForCurrentUser.path
                    }

                    var isDir: ObjCBool = false
                    if FileManager.default.fileExists(atPath: targetDir, isDirectory: &isDir)
                        && isDir.boolValue
                    {
                        persistentCWD = targetDir
                        // Need to return a result here, bypassing performAction which is for MacosUseSDK
                        let resultString = "Changed directory to \(persistentCWD)"
                        return .init(content: [.text(resultString)], isError: false)
                    } else {
                        return .init(
                            content: [.text("cd: \(targetDir): No such file or directory")],
                            isError: true)
                    }
                } else {
                    // Run normal command
                    let output = runShellCommand(command)
                    return .init(content: [.text(output)], isError: false)
                }

            case screenshotTool.name, screenshotAliasTool.name:
                guard let image = captureMainDisplay() else {
                    return .init(content: [.text("Failed to capture screen")], isError: true)
                }
                guard let base64 = encodeBase64JPEG(image: image) else {
                    return .init(content: [.text("Failed to encode screenshot")], isError: true)
                }
                return .init(content: [.text(base64)], isError: false)

            case visionTool.name, ocrAliasTool.name, analyzeAliasTool.name:
                guard let image = captureMainDisplay() else {
                    return .init(
                        content: [.text("Failed to capture screen for analysis")], isError: true)
                }
                let elements = performOCR(on: image)
                guard let jsonString = serializeToJsonString(elements) else {
                    return .init(
                        content: [.text("Failed to serialize vision results")], isError: true)
                }
                return .init(content: [.text(jsonString)], isError: false)

            case scrollTool.name:
                let direction = try getRequiredString(from: params.arguments, key: "direction")
                let amount = (params.arguments?["amount"] as? NSNumber)?.intValue ?? 3

                // Native CGEvent scroll
                let dy =
                    direction == "down"
                    ? Int32(amount * 10) : (direction == "up" ? Int32(-amount * 10) : 0)
                let dx =
                    direction == "right"
                    ? Int32(amount * 10) : (direction == "left" ? Int32(-amount * 10) : 0)
                let scrollEvent = CGEvent(
                    scrollWheelEvent2Source: nil, units: .line, wheelCount: 2, wheel1: dy,
                    wheel2: dx, wheel3: 0)
                scrollEvent?.post(tap: .cghidEventTap)

                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid

            case rightClickTool.name:
                let x = try getRequiredDouble(from: params.arguments, key: "x")
                let y = try getRequiredDouble(from: params.arguments, key: "y")

                // Native right click
                let point = CGPoint(x: x, y: y)
                let mouseDown = CGEvent(
                    mouseEventSource: nil, mouseType: .rightMouseDown, mouseCursorPosition: point,
                    mouseButton: .right)
                let mouseUp = CGEvent(
                    mouseEventSource: nil, mouseType: .rightMouseUp, mouseCursorPosition: point,
                    mouseButton: .right)
                mouseDown?.post(tap: .cghidEventTap)
                mouseUp?.post(tap: .cghidEventTap)

                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid

            case doubleClickTool.name:
                let x = try getRequiredDouble(from: params.arguments, key: "x")
                let y = try getRequiredDouble(from: params.arguments, key: "y")

                // Native double click using CGEvent
                let point = CGPoint(x: x, y: y)
                let mouseDown1 = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: point,
                    mouseButton: .left)
                let mouseUp1 = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: point,
                    mouseButton: .left)
                let mouseDown2 = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: point,
                    mouseButton: .left)
                let mouseUp2 = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: point,
                    mouseButton: .left)

                mouseDown1?.setIntegerValueField(.mouseEventClickState, value: 1)
                mouseUp1?.setIntegerValueField(.mouseEventClickState, value: 1)
                mouseDown2?.setIntegerValueField(.mouseEventClickState, value: 2)
                mouseUp2?.setIntegerValueField(.mouseEventClickState, value: 2)

                mouseDown1?.post(tap: .cghidEventTap)
                mouseUp1?.post(tap: .cghidEventTap)
                mouseDown2?.post(tap: .cghidEventTap)
                mouseUp2?.post(tap: .cghidEventTap)

                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid

            case dragDropTool.name:
                let startX = try getRequiredDouble(from: params.arguments, key: "startX")
                let startY = try getRequiredDouble(from: params.arguments, key: "startY")
                let endX = try getRequiredDouble(from: params.arguments, key: "endX")
                let endY = try getRequiredDouble(from: params.arguments, key: "endY")

                let start = CGPoint(x: startX, y: startY)
                let end = CGPoint(x: endX, y: endY)

                let mouseDown = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: start,
                    mouseButton: .left)
                let mouseDrag = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseDragged, mouseCursorPosition: end,
                    mouseButton: .left)
                let mouseUp = CGEvent(
                    mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: end,
                    mouseButton: .left)

                mouseDown?.post(tap: .cghidEventTap)
                try? await Task.sleep(nanoseconds: 100_000_000)  // Wait for app to catch up
                mouseDrag?.post(tap: .cghidEventTap)
                try? await Task.sleep(nanoseconds: 100_000_000)
                mouseUp?.post(tap: .cghidEventTap)

                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid

            case windowMgmtTool.name:
                let action = try getRequiredString(from: params.arguments, key: "action")

                let appRef = AXUIElementCreateApplication(pid_t(convertedPid))
                var windowValue: AnyObject?
                let result = AXUIElementCopyAttributeValue(
                    appRef, kAXFocusedWindowAttribute as CFString, &windowValue)

                if result == .success, let window = windowValue as! AXUIElement? {
                    switch action {
                    case "minimize":
                        AXUIElementSetAttributeValue(
                            window, kAXMinimizedAttribute as CFString, kCFBooleanTrue)
                    case "maximize":
                        AXUIElementSetAttributeValue(
                            window, kAXMainAttribute as CFString, kCFBooleanTrue)
                    case "make_front":
                        let app = NSRunningApplication(processIdentifier: pid_t(convertedPid))
                        app?.activate(options: .activateIgnoringOtherApps)
                    case "move":
                        let x = try getRequiredDouble(from: params.arguments, key: "x")
                        let y = try getRequiredDouble(from: params.arguments, key: "y")
                        var point = CGPoint(x: x, y: y)
                        if let value = AXValueCreate(.cgPoint, &point) {
                            AXUIElementSetAttributeValue(
                                window, kAXPositionAttribute as CFString, value)
                        }
                    case "resize":
                        let w = try getRequiredDouble(from: params.arguments, key: "width")
                        let h = try getRequiredDouble(from: params.arguments, key: "height")
                        var size = CGSize(width: w, height: h)
                        if let value = AXValueCreate(.cgSize, &size) {
                            AXUIElementSetAttributeValue(
                                window, kAXSizeAttribute as CFString, value)
                        }
                    default:
                        break
                    }

                    // After action, get actual values
                    var actualPos: AnyObject?
                    var actualSize: AnyObject?
                    AXUIElementCopyAttributeValue(
                        window, kAXPositionAttribute as CFString, &actualPos)
                    AXUIElementCopyAttributeValue(window, kAXSizeAttribute as CFString, &actualSize)

                    var pos = CGPoint.zero
                    var sz = CGSize.zero
                    if let pVal = actualPos as! AXValue? { AXValueGetValue(pVal, .cgPoint, &pos) }
                    if let sVal = actualSize as! AXValue? { AXValueGetValue(sVal, .cgSize, &sz) }

                    let resultData = WindowActionResult(
                        action: action,
                        pid: Int(convertedPid),
                        actualX: Double(pos.x),
                        actualY: Double(pos.y),
                        actualWidth: Double(sz.width),
                        actualHeight: Double(sz.height),
                        note: "Window dimensions might be constrained by the application."
                    )

                    if let json = serializeToJsonString(resultData) {
                        return .init(content: [.text(json)], isError: false)
                    }
                }

                primaryAction = .traverseOnly
                options.pidForTraversal = convertedPid

            case setClipboardTool.name:
                let text = try getRequiredString(from: params.arguments, key: "text")
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                return .init(content: [.text("Clipboard updated.")], isError: false)

            case getClipboardTool.name:
                let text = NSPasteboard.general.string(forType: .string) ?? ""
                return .init(content: [.text(text)], isError: false)

            case mediaControlTool.name:
                let action = try getRequiredString(from: params.arguments, key: "action")
                // Using AppleScript for simple system controls as it's most robust for media
                var script = ""
                switch action {
                case "play_pause": script = "tell application \"System Events\" to key code 103"  // Media Play/Pause
                case "next": script = "tell application \"System Events\" to key code 111"
                case "previous": script = "tell application \"System Events\" to key code 101"
                case "volume_up":
                    script =
                        "set volume output volume ((output volume of (get volume settings)) + 10)"
                case "volume_down":
                    script =
                        "set volume output volume ((output volume of (get volume settings)) - 10)"
                case "mute": script = "set volume with output muted"
                case "brightness_up": script = "tell application \"System Events\" to key code 144"
                case "brightness_down":
                    script = "tell application \"System Events\" to key code 145"
                default: break
                }

                if !script.isEmpty {
                    let osascript = Process()
                    osascript.launchPath = "/usr/bin/osascript"
                    osascript.arguments = ["-e", script]
                    osascript.launch()
                    return .init(
                        content: [.text("Executed system control: \(action)")], isError: false)
                }
                return .init(content: [.text("Unknown action")], isError: true)

            default:
                fputs(
                    "error: handler(CallTool): received request for unknown or unsupported tool: \(params.name)\n",
                    stderr)
                throw MCPError.methodNotFound(params.name)
            }

            fputs("log: handler(CallTool): constructed PrimaryAction: \(primaryAction)\n", stderr)

            // --- Execute the Action using MacosUseSDK ---
            let actionResult: ActionResult = await Task { @MainActor in
                fputs(
                    "log: handler(CallTool): executing performAction on MainActor via Task...\n",
                    stderr)
                return await performAction(action: primaryAction, optionsInput: options)
            }.value
            fputs("log: handler(CallTool): performAction task completed.\n", stderr)

            // --- Serialize the ActionResult to JSON ---
            guard let resultJsonString = serializeToJsonString(actionResult) else {
                fputs(
                    "error: handler(CallTool): failed to serialize ActionResult to JSON for tool \(params.name).\n",
                    stderr)
                throw MCPError.internalError("failed to serialize ActionResult to JSON")
            }
            fputs(
                "log: handler(CallTool): successfully serialized ActionResult to JSON string:\n\(resultJsonString)\n",
                stderr)

            // --- Determine if it was an error overall ---
            let isError =
                actionResult.primaryActionError != nil
                || (options.traverseBefore && actionResult.traversalBeforeError != nil)
                || (options.traverseAfter && actionResult.traversalAfterError != nil)

            if isError {
                fputs(
                    "warning: handler(CallTool): Action resulted in an error state (primary: \(actionResult.primaryActionError ?? "nil"), before: \(actionResult.traversalBeforeError ?? "nil"), after: \(actionResult.traversalAfterError ?? "nil")).\n",
                    stderr)
            }

            // --- Return the JSON result ---
            let content: [Tool.Content] = [.text(resultJsonString)]
            return .init(content: content, isError: isError)

        } catch let error as MCPError {
            fputs(
                "error: handler(CallTool): MCPError occurred processing MCP params for tool '\(params.name)': \(error)\n",
                stderr)
            return .init(
                content: [
                    .text(
                        "Error processing parameters for tool '\(params.name)': \(error.localizedDescription)"
                    )
                ], isError: true)
        } catch {
            fputs(
                "error: handler(CallTool): Unexpected error occurred setting up call for tool '\(params.name)': \(error)\n",
                stderr)
            return .init(
                content: [
                    .text(
                        "Unexpected setup error executing tool '\(params.name)': \(error.localizedDescription)"
                    )
                ], isError: true)
        }
    }
    fputs("log: setupAndStartServer: registered CallTool handler.\n", stderr)

    // --- Transport and Start ---
    let transport = StdioTransport()
    fputs("log: setupAndStartServer: created StdioTransport.\n", stderr)

    fputs("log: setupAndStartServer: calling server.start()...\n", stderr)
    try await server.start(transport: transport)
    fputs(
        "log: setupAndStartServer: server.start() completed (background task launched).\n", stderr)

    fputs("log: setupAndStartServer: returning server instance.\n", stderr)
    return server
}

// --- @main Entry Point ---
@main
struct MCPServer {
    // Main entry point - Async
    static func main() async {
        fputs("log: main: starting server (async).\n", stderr)

        // Configure logging if needed (optional)
        // LoggingSystem.bootstrap { label in MultiplexLogHandler([...]) }

        let server: Server
        do {
            fputs("log: main: calling setupAndStartServer()...\n", stderr)
            server = try await setupAndStartServer()
            fputs(
                "log: main: setupAndStartServer() successful, server instance obtained.\n", stderr)

            fputs("log: main: server started, calling server.waitUntilCompleted()...\n", stderr)
            await server.waitUntilCompleted()  // Waits until the server loop finishes/errors
            fputs("log: main: server.waitUntilCompleted() returned. Server has stopped.\n", stderr)

        } catch {
            fputs("error: main: server setup or run failed: \(error)\n", stderr)
            if let mcpError = error as? MCPError {
                fputs("error: main: MCPError details: \(mcpError.localizedDescription)\n", stderr)
            }
            // Consider more specific exit codes if useful
            exit(1)  // Exit with error code
        }

        fputs("log: main: Server processing finished gracefully. Exiting.\n", stderr)
        exit(0)  // Exit cleanly
    }
}
