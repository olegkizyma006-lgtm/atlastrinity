# mcp-server-macos-use

Model Context Protocol (MCP) server in Swift. It allows controlling macOS applications by leveraging the accessibility APIs, primarily through the `MacosUseSDK`.

You can use it in Claude Desktop or other compatible MCP-client.

The server listens for MCP commands over standard input/output (`stdio`) and exposes several tools to interact with applications.

https://github.com/user-attachments/assets/b43622a3-3d20-4026-b02f-e9add06afe2b

## Complete List of Tools

The server provides a comprehensive set of tools for macOS automation, categorized below:

### 1. Application Management & Accessibility

- **`macos-use_open_application_and_traverse`**: Opens/activates an app and traverses its UI tree.
- **`macos-use_refresh_traversal`**: Re-scans the current app's UI tree without action.

### 2. Mouse & Keyboard Control

- **`macos-use_click_and_traverse`**: Left click at (x, y). Supports visual feedback.
- **`macos-use_type_and_traverse`**: Type text into the focused element.
- **`macos-use_press_key_and_traverse`**: Press specific keys (e.g., 'Return', 'Esc') with modifiers.
- **`macos-use_right_click_and_traverse`**: Right click (context menu).
- **`macos-use_double_click_and_traverse`**: Double click.
- **`macos-use_drag_and_drop_and_traverse`**: Drag from (x1, y1) to (x2, y2).
- **`macos-use_scroll_and_traverse`**: Scroll (up/down/left/right).

### 3. Window & System Management

- **`macos-use_window_management`**: Move, resize, minimize, maximize, or focus windows.
- **`macos-use_system_control`**: Media controls (volume, brightness, play/pause).
- **`macos-use_set_clipboard`**: Set clipboard text.
- **`macos-use_get_clipboard`**: Get clipboard text.
- **`macos-use_take_screenshot`**: Capture main screen (Base64 PNG).
- **`macos-use_analyze_screen`**: Vision/OCR analysis of the screen content.

### 4. Native OS Integrations (Universal)

- **`macos-use_get_time`**: Get system time (supports timezones).
- **`macos-use_fetch_url`**: Fetch and parse website content (HTML -> Markdown).
- **`macos-use_run_applescript`**: Execute arbitrary AppleScript code.
- **`macos-use_spotlight_search`**: Fast file search using mdfind.
- **`macos-use_send_notification`**: Send primitive system notifications.

### 5. Productivity Apps

- **Calendar**:
  - `macos-use_calendar_events`: List events.
  - `macos-use_create_event`: Create new events.
- **Reminders**:
  - `macos-use_reminders`: List incomplete reminders.
  - `macos-use_create_reminder`: Create tasks.
- **Notes** (AppleScript-backed):
  - `macos-use_notes_list_folders`: List folders.
  - `macos-use_notes_create_note`: Create notes (HTML supported).
  - `macos-use_notes_get_content`: Read note content.
- **Mail** (AppleScript-backed):
  - `macos-use_mail_send`: Send emails.
  - `macos-use_mail_read_inbox`: Read recent subjects/senders.

### 6. Dynamic Discovery

- **`macos-use_list_tools_dynamic`**: Returns a JSON description of all available tools and their schemas. useful for agents to self-discover capabilities.

## Terminal Command Execution

The server also includes a **`terminal`** (or `execute_command`) tool that allows running low-level shell commands (`/bin/zsh`).

- **Features**: Maintains a persistent Current Working Directory (CWD).
- **Usage**: `{"command": "ls -la"}`
- **Safety**: Be careful with destructive commands like `rm`.

## Common Options

All UI interaction tools accept these optional parameters:

- `showAnimation` (bool): Show a green indicator where the click happens.
- `animationDuration` (float): Speed of the animation.
- `onlyVisibleElements` (bool): Filter out hidden UI nodes.

## Dependencies

- `MacosUseSDK` (Assumed local or external Swift package providing macOS control functionality)

## Building and Running

```bash
# Production build
swift build -c release

# Run
./.build/release/mcp-server-macos-use
```

## Privacy & Permissions

On first run, macOS will prompt for:

- **Accessibility**: Required for UI control.
- **Screen Recording**: Required for screenshots/Vision.
- **Calendar/Reminders**: Required for productivity tools.
- **Apple Events**: Required for AppleScript (controlling Notes/Mail).

**Integrating with Clients (Example: Claude Desktop)**

Once built, you need to tell your client application where to find the server executable. For example, to configure Claude Desktop, you might add the following to its configuration:

```json
{
  "mcpServers": {
    "mcp-server-macos-use": {
      "command": "/path/to/your/project/mcp-server-macos-use/.build/debug/mcp-server-macos-use"
    }
  }
}
```

_Replace `/path/to/your/project/` with the actual absolute path to your `mcp-server-macos-use` directory._

## Help

Reach out to matt@mediar.ai
Discord: m13v\_

## Plans

Happy to tailor the server for your needs, feel free to open an issue or reach out
