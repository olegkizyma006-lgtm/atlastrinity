"""
MCP Registry - Unified Tool Descriptions and Schemas

Single source of truth for all MCP server definitions, tool schemas,
and documentation. Used by all agents (Atlas, Tetyana, Grisha).

This module eliminates duplication across:
- src/brain/prompts/common.py (DEFAULT_REALM_CATALOG, VIBE_TOOLS_DOCUMENTATION)
- src/brain/agents/tetyana.py (MACOS_USE_SCHEMAS)
"""

from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
#                           SERVER CATALOG
# ═══════════════════════════════════════════════════════════════════════════════
# Complete catalog of all MCP servers with detailed descriptions.
# Used by LLM to decide which servers to initialize (lazy loading).

SERVER_CATALOG: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1 - CORE SERVERS (Always prioritize)
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use": {
        "name": "macos-use",
        "tier": 1,
        "category": "core",
        "description": "Universal macOS Commander (Swift binary - 39+ tools)",
        "capabilities": [
            "GUI automation (click, type, scroll, drag, window management)",
            "Terminal/shell command execution",
            "Screenshots and Vision OCR (Apple Vision Framework)",
            "Calendar events management",
            "Reminders management",
            "Notes (create, read, list folders)",
            "Mail (send, read inbox)",
            "Finder operations (list, open, trash)",
            "System control (volume, brightness, media)",
            "Fetch URLs to markdown",
            "Get system time with timezone",
            "AppleScript execution",
            "Spotlight search",
            "System notifications",
        ],
        "key_tools": [
            "macos-use_open_application_and_traverse",
            "macos-use_click_and_traverse",
            "macos-use_type_and_traverse",
            "macos-use_press_key_and_traverse",
            "execute_command",
            "macos-use_take_screenshot",
            "macos-use_analyze_screen",
            "macos-use_calendar_events",
            "macos-use_create_event",
            "macos-use_reminders",
            "macos-use_notes_create_note",
            "macos-use_mail_send",
            "macos-use_fetch_url",
            "macos-use_get_time",
            "macos-use_list_tools_dynamic",
        ],
        "when_to_use": "ANY computer interaction, macOS GUI automation, terminal commands, Apple productivity apps",
        "priority_note": "ALWAYS prefer macos-use for GUI, terminal, fetch, time over other servers",
        "protocol_note": "DISCOVERY FIRST POLICY: Always call 'macos-use_list_tools_dynamic' (discovery) before starting a complex task to get the latest 39+ tool schemas and instructions directly from the server. This ensures 100% precision.",
    },
    "filesystem": {
        "name": "filesystem",
        "tier": 1,
        "category": "core",
        "description": "File operations (read, write, list, search)",
        "capabilities": [
            "Read file contents",
            "Write/create files",
            "List directory contents",
            "Search files by name/content",
            "Get file metadata",
        ],
        "key_tools": [
            "read_file",
            "write_file",
            "list_directory",
            "search_files",
        ],
        "when_to_use": "File operations within home directory. For paths outside ~, use macos-use.execute_command",
        "restrictions": "Limited to home directory only",
    },
    "sequential-thinking": {
        "name": "sequential-thinking",
        "tier": 1,
        "category": "core",
        "description": "Step-by-step reasoning for complex decisions and risk analysis",
        "capabilities": [
            "Deep logical reasoning",
            "Risk assessment",
            "Multi-step problem decomposition",
            "Consequence simulation",
        ],
        "key_tools": ["sequentialthinking"],
        "when_to_use": "Complex decisions, dangerous operations (rm -rf), multi-step logic, hypothesis testing",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2 - HIGH PRIORITY SERVERS
    # ─────────────────────────────────────────────────────────────────────────

    "memory": {
        "name": "memory",
        "tier": 2,
        "category": "knowledge",
        "description": "Knowledge graph for persistent entity storage",
        "capabilities": [
            "Create entities",
            "Create relations between entities",
            "Search nodes",
            "Retrieve entity observations",
        ],
        "key_tools": ["create_entities", "create_relations", "search_nodes"],
        "when_to_use": "Storing/retrieving long-term knowledge, entity relationships",
    },

    "vibe": {
        "name": "vibe",
        "tier": 2,
        "category": "ai",
        "description": "AI-powered CODING and DEBUGGING engine (Mistral CLI)",
        "capabilities": [
            "Deep error analysis with auto-fix",
            "Code review",
            "Writing complex software/scripts",
            "Self-healing when tools fail",
        ],
        "key_tools": [
            "vibe_prompt",
            "vibe_analyze_error",
            "vibe_implement_feature",
            "vibe_code_review",
        ],
        "when_to_use": "ONLY for: 1) Writing code/scripts (software dev), 2) Fixing hard errors (self-healing). DO NOT use for general planning or simple file tasks.",
    },

    "system": {
        "name": "system",
        "tier": 1,
        "category": "core",
        "description": "Internal Trinity System tools",
        "capabilities": [
            "Restarting MCP servers",
            "System status management"
        ],
        "key_tools": ["restart_mcp_server"],
        "when_to_use": "Use when an MCP server is unresponsive, returns 'Connection closed', or needs a fresh start.",
    },

    "puppeteer": {
        "name": "puppeteer",
        "tier": 2,
        "category": "browser",
        "description": "High-speed Headless Browser (Puppeteer)",
        "capabilities": [
            "Web searching (Google/Bing) without API keys",
            "Navigating complex URLs and SPA applications",
            "Interacting with elements (click, type, fill)",
            "Capturing full-page screenshots",
            "Executing JavaScript on pages",
        ],
        "key_tools": ["puppeteer_navigate", "puppeteer_screenshot", "puppeteer_click"],
        "when_to_use": "ANY web task: searching for info, checking weather, scraping data, or interacting with websites.",
    },
    "chrome-devtools": {
        "name": "chrome-devtools",
        "tier": 2,
        "category": "browser",
        "description": "Chrome DevTools Protocol for browser automation",
        "capabilities": [
            "Low-level browser control",
            "Network monitoring",
            "Console log access",
            "DOM inspection",
        ],
        "key_tools": ["captureScreenshot", "console_logs", "network_requests"],
        "when_to_use": "Advanced browser debugging or when puppeteer is insufficient.",
    },
    "graph": {
        "name": "graph",
        "tier": 2,
        "category": "visualization",
        "description": "Knowledge Graph visualization and export",
        "capabilities": [
            "Generate Mermaid diagrams",
            "Visualize task dependencies",
            "Export graph data",
        ],
        "key_tools": ["generate_mermaid", "export_graph"],
        "when_to_use": "Visualizing complex system states, task flows, or knowledge structures.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 3-4 - OPTIONAL SERVERS
    # ─────────────────────────────────────────────────────────────────────────

}

# ═══════════════════════════════════════════════════════════════════════════════
#                           TOOL SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════
# Exact parameter schemas for tool validation and argument handling.
# Previously hardcoded in tetyana.py as MACOS_USE_SCHEMAS.

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: GUI Automation
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_open_application_and_traverse": {
        "server": "macos-use",
        "required": ["identifier"],
        "optional": [],
        "types": {"identifier": str},
        "description": "Open application by name, path, or bundleID",
    },
    "macos-use_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Click at coordinates (x, y)",
    },
    "macos-use_type_and_traverse": {
        "server": "macos-use",
        "required": ["text"],
        "optional": ["pid", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "text": str,
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Type text into focused element",
    },
    "macos-use_press_key_and_traverse": {
        "server": "macos-use",
        "required": ["keyName"],
        "optional": ["pid", "modifierFlags", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "keyName": str,
            "modifierFlags": list,
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Press key with optional modifiers (Return, Tab, Escape, shortcuts)",
    },
    "macos-use_refresh_traversal": {
        "server": "macos-use",
        "required": [],
        "optional": ["pid"],
        "types": {"pid": int},
        "description": "Force refresh UI accessibility tree",
    },
    "macos-use_scroll_and_traverse": {
        "server": "macos-use",
        "required": ["direction"],
        "optional": ["pid", "amount", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "direction": str,
            "amount": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Scroll in direction (up, down, left, right)",
    },
    "macos-use_right_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Right-click (context menu) at coordinates",
    },
    "macos-use_double_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Double-click at coordinates",
    },
    "macos-use_drag_and_drop_and_traverse": {
        "server": "macos-use",
        "required": ["startX", "startY", "endX", "endY"],
        "optional": ["pid", "showAnimation", "animationDuration"],
        "types": {
            "pid": int,
            "startX": (int, float),
            "startY": (int, float),
            "endX": (int, float),
            "endY": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Drag from start to end coordinates",
    },
    "macos-use_window_management": {
        "server": "macos-use",
        "required": ["action"],
        "optional": ["pid", "x", "y", "width", "height"],
        "types": {
            "pid": int,
            "action": str,
            "x": (int, float),
            "y": (int, float),
            "width": (int, float),
            "height": (int, float),
        },
        "description": "Window management (move, resize, minimize, maximize, make_front)",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Terminal
    # ─────────────────────────────────────────────────────────────────────────
    "execute_command": {
        "server": "macos-use",
        "required": ["command"],
        "optional": [],
        "types": {"command": str},
        "description": "Execute shell command (PRIMARY terminal access)",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Screenshots & Vision
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_take_screenshot": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "Capture screen as Base64 PNG",
    },
    "macos-use_analyze_screen": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "Apple Vision OCR - analyze screen text",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Clipboard & System
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_set_clipboard": {
        "server": "macos-use",
        "required": ["text"],
        "optional": ["showAnimation", "animationDuration"],
        "types": {
            "text": str,
            "showAnimation": bool,
            "animationDuration": (int, float),
        },
        "description": "Set clipboard content",
    },
    "macos-use_get_clipboard": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "Get clipboard content",
    },
    "macos-use_system_control": {
        "server": "macos-use",
        "required": ["action"],
        "optional": [],
        "types": {"action": str},
        "description": "System control (play_pause, volume_up/down, mute, brightness)",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Productivity Apps
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_calendar_events": {
        "server": "macos-use",
        "required": [],
        "optional": ["days_ahead"],
        "types": {"days_ahead": int},
        "description": "Get calendar events",
    },
    "macos-use_create_event": {
        "server": "macos-use",
        "required": ["title", "start_date", "end_date"],
        "optional": ["notes", "calendar_name"],
        "types": {
            "title": str,
            "start_date": str,
            "end_date": str,
            "notes": str,
            "calendar_name": str,
        },
        "description": "Create calendar event",
    },
    "macos-use_reminders": {
        "server": "macos-use",
        "required": [],
        "optional": ["list_name"],
        "types": {"list_name": str},
        "description": "Get reminders",
    },
    "macos-use_create_reminder": {
        "server": "macos-use",
        "required": ["title"],
        "optional": ["list_name", "due_date", "notes"],
        "types": {"title": str, "list_name": str, "due_date": str, "notes": str},
        "description": "Create reminder",
    },
    "macos-use_notes_list_folders": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "List Notes folders",
    },
    "macos-use_notes_create_note": {
        "server": "macos-use",
        "required": ["title"],
        "optional": ["body", "folder"],
        "types": {"title": str, "body": str, "folder": str},
        "description": "Create Apple Note",
    },
    "macos-use_notes_get_content": {
        "server": "macos-use",
        "required": ["note_id"],
        "optional": [],
        "types": {"note_id": str},
        "description": "Get note content by ID",
    },
    "macos-use_mail_send": {
        "server": "macos-use",
        "required": ["to", "subject", "body"],
        "optional": ["from_account"],
        "types": {"to": str, "subject": str, "body": str, "from_account": str},
        "description": "Send email via Apple Mail",
    },
    "macos-use_mail_read_inbox": {
        "server": "macos-use",
        "required": [],
        "optional": ["limit"],
        "types": {"limit": int},
        "description": "Read inbox messages",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Finder
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_finder_list_files": {
        "server": "macos-use",
        "required": [],
        "optional": ["path"],
        "types": {"path": str},
        "description": "List files in Finder",
    },
    "macos-use_finder_get_selection": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "Get selected items in Finder",
    },
    "macos-use_finder_open_path": {
        "server": "macos-use",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Open path in Finder",
    },
    "macos-use_finder_move_to_trash": {
        "server": "macos-use",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Move item to Trash",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: Utilities
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_fetch_url": {
        "server": "macos-use",
        "required": ["url"],
        "optional": ["timeout"],
        "types": {"url": str, "timeout": int},
        "description": "Fetch URL content as markdown",
    },
    "macos-use_get_time": {
        "server": "macos-use",
        "required": [],
        "optional": ["timezone"],
        "types": {"timezone": str},
        "description": "Get system time with timezone",
    },
    "macos-use_run_applescript": {
        "server": "macos-use",
        "required": ["script"],
        "optional": [],
        "types": {"script": str},
        "description": "Execute AppleScript",
    },
    "macos-use_spotlight_search": {
        "server": "macos-use",
        "required": ["query"],
        "optional": ["scope"],
        "types": {"query": str, "scope": str},
        "description": "Spotlight file search",
    },
    "macos-use_send_notification": {
        "server": "macos-use",
        "required": ["title"],
        "optional": ["subtitle", "body"],
        "types": {"title": str, "subtitle": str, "body": str},
        "description": "Show system notification",
    },
    "macos-use_list_tools_dynamic": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "types": {},
        "description": "ULTIMATE DISCOVERY TOOL: Get the complete, real-time list of ALL 39+ available tools and their schemas from the macos-use server. Use this if a tool seems missing or you need to verify current capabilities.",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # FILESYSTEM: File Operations
    # ─────────────────────────────────────────────────────────────────────────
    "read_file": {
        "server": "filesystem",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Read file contents (home directory only)",
    },
    "write_file": {
        "server": "filesystem",
        "required": ["path", "content"],
        "optional": [],
        "types": {"path": str, "content": str},
        "description": "Write/create file (home directory only)",
    },
    "list_directory": {
        "server": "filesystem",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "List directory contents",
    },
    "search_files": {
        "server": "filesystem",
        "required": ["path", "pattern"],
        "optional": [],
        "types": {"path": str, "pattern": str},
        "description": "Search files by name or content",
    },
    "get_file_info": {
        "server": "filesystem",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Get file metadata",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY: Knowledge Graph
    # ─────────────────────────────────────────────────────────────────────────
    "create_entities": {
        "server": "memory",
        "required": ["entities"],
        "optional": [],
        "types": {"entities": list},
        "description": "Create entities in knowledge graph",
    },
    "create_relations": {
        "server": "memory",
        "required": ["relations"],
        "optional": [],
        "types": {"relations": list},
        "description": "Create relations between entities",
    },
    "add_observations": {
        "server": "memory",
        "required": ["observations"],
        "optional": [],
        "types": {"observations": list},
        "description": "Add observations to entities",
    },
    "search_nodes": {
        "server": "memory",
        "required": ["query"],
        "optional": [],
        "types": {"query": str},
        "description": "Search entities in knowledge graph",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # SEQUENTIAL THINKING: Reasoning
    # ─────────────────────────────────────────────────────────────────────────
    "sequentialthinking": {
        "server": "sequential-thinking",
        "required": ["thought", "thoughtNumber", "totalThoughts"],
        "optional": ["nextThoughtNeeded"],
        "types": {
            "thought": str,
            "thoughtNumber": int,
            "totalThoughts": int,
            "nextThoughtNeeded": bool,
        },
        "description": "Multi-step sequential reasoning",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # VIBE: AI-Powered Tools
    # ─────────────────────────────────────────────────────────────────────────
    "vibe_prompt": {
        "server": "vibe",
        "required": ["prompt"],
        "optional": ["cwd", "timeout_s", "output_format", "auto_approve", "max_turns", "max_price", "resume"],
        "types": {
            "prompt": str,
            "cwd": str,
            "timeout_s": (int, float),
            "output_format": str,
            "auto_approve": bool,
            "max_turns": int,
            "max_price": (int, float),
            "resume": str,
        },
        "description": "Send prompt to Vibe AI",
    },
    "vibe_analyze_error": {
        "server": "vibe",
        "required": ["error_message"],
        "optional": ["log_context", "recovery_history", "file_path", "cwd", "timeout_s", "auto_fix"],
        "types": {
            "error_message": str,
            "log_context": str,
            "recovery_history": str,
            "file_path": str,
            "cwd": str,
            "timeout_s": (int, float),
            "auto_fix": bool,
        },
        "description": "Deep error analysis with optional auto-fix",
    },
    "vibe_code_review": {
        "server": "vibe",
        "required": ["file_path"],
        "optional": ["focus_areas", "cwd", "timeout_s"],
        "types": {
            "file_path": str,
            "focus_areas": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "AI code review",
    },
    "vibe_implement_feature": {
        "server": "vibe",
        "required": ["goal"],
        "optional": ["context_files", "constraints", "cwd", "timeout_s"],
        "types": {
            "goal": str,
            "context_files": list,
            "constraints": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "DEEP CODING: Plan and implement complex features across multiple files",
    },
    "vibe_smart_plan": {
        "server": "vibe",
        "required": ["objective"],
        "optional": ["context", "cwd", "timeout_s"],
        "types": {
            "objective": str,
            "context": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "Generate execution plan for objective",
    },
    "vibe_ask": {
        "server": "vibe",
        "required": ["question"],
        "optional": ["cwd", "timeout_s"],
        "types": {"question": str, "cwd": str, "timeout_s": (int, float)},
        "description": "Quick read-only question (no file changes)",
    },
    "vibe_list_sessions": {
        "server": "vibe",
        "required": [],
        "optional": ["limit"],
        "types": {"limit": int},
        "description": "List Vibe sessions",
    },
    "vibe_session_details": {
        "server": "vibe",
        "required": ["session_id_or_file"],
        "optional": [],
        "types": {"session_id_or_file": str},
        "description": "Get session details",
    },
    "vibe_execute_subcommand": {
        "server": "vibe",
        "required": ["subcommand"],
        "optional": ["args", "cwd", "timeout_s", "env"],
        "types": {
            "subcommand": str,
            "args": list,
            "cwd": str,
            "timeout_s": (int, float),
            "env": dict,
        },
        "description": "Execute Vibe CLI subcommand",
    },
    "vibe_which": {
        "server": "vibe",
        "required": [],
        "optional": [],
        "types": {},
        "description": "Check Vibe CLI installation",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # BROWSER: Puppeteer
    # ─────────────────────────────────────────────────────────────────────────
    "puppeteer_navigate": {
        "server": "puppeteer",
        "required": ["url"],
        "optional": [],
        "types": {"url": str},
        "description": "Navigate to a URL",
    },
    "puppeteer_screenshot": {
        "server": "puppeteer",
        "required": ["name"],
        "optional": ["width", "height"],
        "types": {"name": str, "width": int, "height": int},
        "description": "Take a screenshot of the current page",
    },
    "puppeteer_click": {
        "server": "puppeteer",
        "required": ["selector"],
        "optional": [],
        "types": {"selector": str},
        "description": "Click an element on the page",
    },
    "puppeteer_fill": {
        "server": "puppeteer",
        "required": ["selector", "value"],
        "optional": [],
        "types": {"selector": str, "value": str},
        "description": "Fill an input field",
    },
    "puppeteer_evaluate": {
        "server": "puppeteer",
        "required": ["script"],
        "optional": [],
        "types": {"script": str},
        "description": "Execute JavaScript on the page",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # TRINITY NATIVE: System Tools
    # ─────────────────────────────────────────────────────────────────────────
    "restart_mcp_server": {
        "server": "_trinity_native",
        "required": ["server_name"],
        "optional": [],
        "types": {"server_name": str},
        "description": "Restart MCP server by name",
    },
    "query_db": {
        "server": "_trinity_native",
        "required": ["query"],
        "optional": ["params"],
        "types": {"query": str, "params": dict},
        "description": "Query internal database",
    },
    # ─────────────────────────────────────────────────────────────────────────
    # ALIASES (for backwards compatibility and LLM flexibility)
    # ─────────────────────────────────────────────────────────────────────────
    "terminal": {"alias_for": "execute_command", "server": "macos-use"},
    "run_command": {"alias_for": "execute_command", "server": "macos-use"},
    "screenshot": {"alias_for": "macos-use_take_screenshot", "server": "macos-use"},
    "vision": {"alias_for": "macos-use_analyze_screen", "server": "macos-use"},
    "ocr": {"alias_for": "macos-use_analyze_screen", "server": "macos-use"},
    "analyze": {"alias_for": "macos-use_analyze_screen", "server": "macos-use"},
    "scroll": {"alias_for": "macos-use_scroll_and_traverse", "server": "macos-use"},
    "right_click": {"alias_for": "macos-use_right_click_and_traverse", "server": "macos-use"},
    "double_click": {"alias_for": "macos-use_double_click_and_traverse", "server": "macos-use"},
    "drag_drop": {"alias_for": "macos-use_drag_and_drop_and_traverse", "server": "macos-use"},
    "window_mgmt": {"alias_for": "macos-use_window_management", "server": "macos-use"},
    "set_clipboard": {"alias_for": "macos-use_set_clipboard", "server": "macos-use"},
    "get_clipboard": {"alias_for": "macos-use_get_clipboard", "server": "macos-use"},
    "system_control": {"alias_for": "macos-use_system_control", "server": "macos-use"},
    "fetch_url": {"alias_for": "macos-use_fetch_url", "server": "macos-use"},
    "get_time": {"alias_for": "macos-use_get_time", "server": "macos-use"},
    "run_applescript": {"alias_for": "macos-use_run_applescript", "server": "macos-use"},
    "directory_tree": {"alias_for": "list_directory", "server": "filesystem"},
    "tree": {"alias_for": "list_directory", "server": "filesystem"},
    "list_dir": {"alias_for": "list_directory", "server": "filesystem"},
    "browser_navigate": {"alias_for": "puppeteer_navigate", "server": "puppeteer"},
    "browser_screenshot": {"alias_for": "puppeteer_screenshot", "server": "puppeteer"},
}


# ═══════════════════════════════════════════════════════════════════════════════
#                    VIBE TOOLS DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

VIBE_DOCUMENTATION = """VIBE MCP SERVER - AI-POWERED DEBUGGING & SELF-HEALING

The 'vibe' server provides access to Mistral AI for advanced debugging, code analysis, and self-healing.
All Vibe operations run in PROGRAMMATIC CLI mode (not interactive TUI) - output is fully visible in logs.

AVAILABLE VIBE TOOLS:

1. **vibe_prompt** (PRIMARY TOOL)
   Purpose: Send any prompt to Vibe AI for analysis or action
   Args:
     - prompt: The message/query (required)
     - cwd: Working directory (optional)
     - timeout_s: Timeout in seconds (default 300)
     - output_format: 'json', 'text', or 'streaming' (default 'json')
     - auto_approve: Auto-approve tool calls (default True)
     - max_turns: Max conversation turns (default 10)
   Example: vibe_prompt(prompt="Why is this code failing?", cwd="/path/to/project")

2. **vibe_analyze_error** (SELF-HEALING)
   Purpose: Deep error analysis with optional auto-fix
   Args:
     - error_message: The error/stack trace (required)
     - log_context: Recent logs for context (optional)
     - file_path: Path to problematic file (optional)
     - auto_fix: Whether to apply fixes (default True)
   Example: vibe_analyze_error(error_message="TypeError: x is undefined", log_context="...", auto_fix=True)

3. **vibe_code_review**
   Purpose: Request AI code review for a file
   Args:
     - file_path: Path to review (required)
     - focus_areas: Areas to focus on, e.g., "security", "performance" (optional)
   Example: vibe_code_review(file_path="/src/main.py", focus_areas="security")

4. **vibe_smart_plan**
   Purpose: Generate execution plan for complex objectives
   Args:
     - objective: The goal to plan for (required)
     - context: Additional context (optional)
   Example: vibe_smart_plan(objective="Implement OAuth2 authentication")

5. **vibe_ask** (READ-ONLY)
   Purpose: Ask a quick question without file modifications
   Args:
     - question: The question (required)
   Example: vibe_ask(question="What's the best way to handle async errors in Python?")

6. **vibe_execute_subcommand**
   Purpose: Execute a specific Vibe CLI subcommand (non-AI utility)
   Args:
     - subcommand: 'list-editors', 'run', 'enable', 'disable', 'install', etc. (required)
     - args: List of string arguments (optional)
     - cwd: Working directory (optional)
   Example: vibe_execute_subcommand(subcommand="list-editors")

7. **vibe_which**
   Purpose: Check Vibe CLI installation path and version
   Example: vibe_which()

TRINITY NATIVE SYSTEM TOOLS (Any Agent):
- `restart_mcp_server(server_name)`: Force restart an MCP server.
- `query_db(query, params)`: Query the internal system database.

WHEN TO USE VIBE:
- When Tetyana/Grisha fail after multiple attempts
- Complex debugging requiring AI reasoning
- Code review before committing
- Planning multi-step implementations
- Understanding unfamiliar code patterns
- System diagnostics

IMPORTANT: All Vibe output is logged and visible in the Electron app logs!
"""


# ═══════════════════════════════════════════════════════════════════════════════
#                           VOICE PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════════

VOICE_PROTOCOL = """VOICE COMMUNICATION PROTOCOL (Text-To-Speech):

Your `voice_message`, `voice_summary`, `voice_response`, and `final_report` fields are PUBLIC.
LANGUAGE: UKRAINIAN ONLY. 
CRITICAL RULE: ZERO ENGLISH WORDS. Do not include English paths (use "домашня папка", "робочий стіл"), tool names, or technical terms. Everything must be localized into high-quality natural Ukrainian.
Even if the internal reasoning is English, the voice content MUST BE 100% Ukrainian.

RULES FOR VOICE CONTEXT:
1. **Be Concise & Specific**: defined "essence" of the action.
   - BAD: "Running ls command in /Users/oleg/Downloads"
   - GOOD: "Перевіряю список файлів у папці завантажень."
   - GOOD: "Помилка доступу. Пробую права адміністратора."

2. **No Hardcodes**: Do not use generic phrases like "Thinking..." or "Step done". Always include context.
   - BAD: "Крок завершено."
   - GOOD: "Сервер запущено на роз'ємі 8000."

3. **Error Reporting**:
   - format: "{Failure essence}. {Reason (short)}. {Next step}."
   - Example: "Не вдалося клонувати репо. Невірний токен. Перевіряю змінні середовища."

4. **Tone**: Professional, Active, Fast-paced. Like a senior engineer reporting to a lead.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#                           UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def get_server_catalog_for_prompt(include_key_tools: bool = True) -> str:
    """
    Generate LLM-readable server catalog for prompts.
    This replaces the hardcoded DEFAULT_REALM_CATALOG in common.py.
    """
    lines = ["AVAILABLE REALMS (MCP Servers):", ""]

    # Group by tier
    by_tier: Dict[int, List[Dict]] = {}
    for name, info in SERVER_CATALOG.items():
        tier = info.get("tier", 4)
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(info)

    tier_names = {
        1: "TIER 1 - CORE",
        2: "TIER 2 - HIGH PRIORITY",
        3: "TIER 3 - OPTIONAL",
        4: "TIER 4 - SPECIALIZED",
    }

    for tier in sorted(by_tier.keys()):
        lines.append(f"{tier_names.get(tier, f'TIER {tier}')}:")

        for server in by_tier[tier]:
            name = server["name"]
            desc = server["description"]
            lines.append(f"- {name}: {desc}")

            if include_key_tools and "key_tools" in server:
                tools = ", ".join(server["key_tools"][:5])
                if len(server["key_tools"]) > 5:
                    tools += ", ..."
                lines.append(f"  Key tools: {tools}")

            if "priority_note" in server:
                lines.append(f"  NOTE: {server['priority_note']}")

        lines.append("")

    lines.append("DEPRECATED (Use macos-use instead):")
    lines.append("- fetch → macos-use_fetch_url")
    lines.append("- time → macos-use_get_time")
    lines.append("- apple-mcp → macos-use Calendar/Reminders/Notes/Mail tools")
    lines.append("- git → macos-use execute_command('git ...')")
    lines.append("- notes → filesystem or macos-use (Apple Notes)")
    lines.append("- search → macos-use chrome or fetch_url")
    lines.append("- docker, postgres, slack → Disabled/Removed")
    lines.append("")
    lines.append("CRITICAL: Do NOT invent high-level tools. Use only the real TOOLS found inside these Realms.")

    return "\n".join(lines)


def get_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get schema for a specific tool.
    Resolves aliases to their canonical form.
    """
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema and "alias_for" in schema:
        return TOOL_SCHEMAS.get(schema["alias_for"])
    return schema


def get_server_for_tool(tool_name: str) -> Optional[str]:
    """Get the server name for a tool."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema:
        if "alias_for" in schema:
            return schema.get("server")
        return schema.get("server")
    return None


def get_servers_for_task(task_type: str) -> List[str]:
    """
    Suggest servers based on task type.
    Used for intelligent lazy initialization.
    """
    task_lower = task_type.lower()

    # Direct mappings
    if any(x in task_lower for x in ["gui", "click", "type", "window", "app", "screen"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["terminal", "command", "shell", "bash"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["file", "read", "write", "directory"]):
        return ["filesystem", "macos-use"]
    if any(x in task_lower for x in ["search", "web", "internet", "google", "find", "browser", "navigate", "automation", "scrape"]):
        return ["puppeteer", "macos-use"]
    if any(x in task_lower for x in ["calendar", "event", "meeting"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["reminder", "todo", "task"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["note", "notes"]):
        return ["macos-use"]  # Use Apple Notes via macos-use
    if any(x in task_lower for x in ["mail", "email"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["git", "commit", "push", "pull", "branch"]):
        return ["macos-use"]  # Route git to macos-use (legacy override)
    if any(x in task_lower for x in ["github", "repository", "issue", "pr"]):
        return ["macos-use"]  # Route github to macos-use (browser/cli)
    if any(x in task_lower for x in ["debug", "error", "fix", "analyze"]):
        return ["vibe", "sequential-thinking"]
    if any(x in task_lower for x in ["code", "review", "refactor"]):
        return ["vibe"]
    if any(x in task_lower for x in ["think", "reason", "complex", "decision"]):
        return ["sequential-thinking"]
    if any(x in task_lower for x in ["time", "clock", "date"]):
        return ["macos-use"]
    if any(x in task_lower for x in ["fetch", "url", "http", "download"]):
        return ["macos-use"]

    # Default: return core servers
    return ["macos-use", "filesystem"]


def get_all_tool_names() -> List[str]:
    """Get list of all available tool names (excluding aliases)."""
    return [name for name, schema in TOOL_SCHEMAS.items() if "alias_for" not in schema]


def get_tool_names_for_server(server_name: str) -> List[str]:
    """Get all tool names for a specific server."""
    return [
        name
        for name, schema in TOOL_SCHEMAS.items()
        if schema.get("server") == server_name and "alias_for" not in schema
    ]
