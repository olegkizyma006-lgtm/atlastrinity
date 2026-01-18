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
        "description": "Universal macOS Commander (Swift binary - 52 tools)",
        "capabilities": [
            "GUI automation (click, type, scroll, drag, window management)",
            "Terminal/shell command execution (execute_command)",
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
        "protocol_note": "DISCOVERY FIRST POLICY: Always call 'macos-use_list_tools_dynamic' before starting a complex task to get the latest 52 tool schemas and instructions directly from the server.",
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
        "when_to_use": "File operations within allowed paths (~ and /tmp). For other paths, use macos-use.execute_command",
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
        "when_to_use": "Complex decisions, dangerous operations, multi-step logic, hypothesis testing",
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
        "when_to_use": "Use when an MCP server is unresponsive or needs a restart.",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2 - HIGH PRIORITY SERVERS
    # ─────────────────────────────────────────────────────────────────────────
    "vibe": {
        "name": "vibe",
        "tier": 2,
        "category": "ai",
        "description": "Mistral AI-powered CODING and DEBUGGING engine (12 tools)",
        "capabilities": [
            "Deep error analysis with auto-fix",
            "Code review and architectural advice",
            "Writing complex software/scripts",
            "Self-healing when tools fail",
            "Direct PostgreSQL database inspection (vibe_check_db)",
            "Context-aware system monitoring (vibe_get_system_context)",
            "Long-running task planning",
        ],
        "key_tools": [
            "vibe_prompt",
            "vibe_analyze_error",
            "vibe_implement_feature",
            "vibe_check_db",
            "vibe_get_system_context",
            "vibe_code_review",
            "vibe_smart_plan",
            "vibe_ask",
        ],
        "when_to_use": "ONLY for software development, writing code/scripts, fixing hard errors (self-healing), and deep system state inspection via DB.",
    },
    "memory": {
        "name": "memory",
        "tier": 2,
        "category": "knowledge",
        "description": "Knowledge Graph-based long-term memory (PostgreSQL + ChromaDB)",
        "capabilities": [
            "Entity and relationship tracking (PostgreSQL)",
            "Observation and fact management",
            "Semantic search across memories (ChromaDB)",
            "Persistent graph-based recall for all agents",
        ],
        "key_tools": [
            "create_entities",
            "add_observations",
            "create_relation",
            "search",
            "get_entity",
            "list_entities",
        ],
        "when_to_use": "Storing or recalling facts, entities, and historical observations about the user, system, or codebase.",
    },
    "graph": {
        "name": "graph",
        "tier": 2,
        "category": "visualization",
        "description": "Knowledge Graph visualization and interactive exploration",
        "capabilities": [
            "JSON data export of memory graph (D3.js compatible)",
            "Mermaid.js flowchart generation with filtering",
            "Deep node inspection and relationship traversal",
        ],
        "key_tools": [
            "generate_mermaid",
            "get_node_details",
            "get_related_nodes",
            "get_graph_json",
        ],
        "when_to_use": "Visualizing or traversing the knowledge graph state to understand complex system or data relationships.",
    },
    "puppeteer": {
        "name": "puppeteer",
        "tier": 2,
        "category": "browser",
        "description": "Headless Browser (Puppeteer) for web automation",
        "capabilities": [
            "Web searching (Google/Bing)",
            "Navigating complex URLs and SPAs",
            "Interacting with elements (click, type, fill)",
            "Capturing page screenshots",
        ],
        "key_tools": ["puppeteer_navigate", "puppeteer_screenshot", "puppeteer_click"],
        "when_to_use": "Web searching, checking weather, scraping data, website interaction.",
    },
    "chrome-devtools": {
        "name": "chrome-devtools",
        "tier": 2,
        "category": "browser",
        "description": "Chrome DevTools Protocol for advanced browser automation",
        "capabilities": [
            "Low-level browser control",
            "Network monitoring",
            "Console log access",
        ],
        "key_tools": ["captureScreenshot", "console_logs", "network_requests"],
        "when_to_use": "Advanced browser debugging or when puppeteer is insufficient.",
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
    # VIBE: AI Coding & Self-Healing
    # ─────────────────────────────────────────────────────────────────────────
    "vibe_prompt": {
        "server": "vibe",
        "required": ["prompt"],
        "optional": ["cwd", "timeout_s", "output_format", "auto_approve", "max_turns", "resume"],
        "types": {
            "prompt": str,
            "cwd": str,
            "timeout_s": (int, float),  # Timeout in seconds (default 3600)
            "output_format": str,
            "auto_approve": bool,
            "max_turns": int,
            "resume": str,
        },
        "description": "Run a general AI coding or analysis task with full TUI capabilities",
    },
    "vibe_analyze_error": {
        "server": "vibe",
        "required": ["error_message"],
        "optional": ["context_files", "cwd", "timeout_s"],
        "types": {
            "error_message": str,
            "context_files": list,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "SELF-HEALING: Analyze an error and automatically implement a fix",
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
    "vibe_code_review": {
        "server": "vibe",
        "required": ["files"],
        "optional": ["focus", "cwd", "timeout_s"],
        "types": {
            "files": list,
            "focus": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "Perform a deep technical code review of specified files",
    },
    "vibe_smart_plan": {
        "server": "vibe",
        "required": ["goal"],
        "optional": ["cwd", "timeout_s"],
        "types": {
            "goal": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "Generate a detailed multi-step execution plan for a complex task",
    },
    "vibe_ask": {
        "server": "vibe",
        "required": ["question"],
        "optional": ["cwd", "timeout_s"],
        "types": {
            "question": str,
            "cwd": str,
            "timeout_s": (int, float),
        },
        "description": "Ask a technical or architectural question without making changes",
    },
    "vibe_execute_subcommand": {
        "server": "vibe",
        "required": ["subcommand"],
        "optional": ["args", "cwd"],
        "types": {
            "subcommand": str,
            "args": list,
            "cwd": str,
        },
        "description": "Execute a raw Vibe CLI subcommand (e.g., list-editors)",
    },
    "vibe_list_sessions": {
        "server": "vibe",
        "required": [],
        "optional": [],
        "description": "List all active and historical Vibe sessions",
    },
    "vibe_session_details": {
        "server": "vibe",
        "required": ["session_id"],
        "optional": [],
        "types": {"session_id": str},
        "description": "Get detailed logs and status for a specific Vibe session",
    },
    "vibe_which": {
        "server": "vibe",
        "required": [],
        "optional": [],
        "description": "Check if Vibe CLI is installed and find its path",
    },
    "vibe_check_db": {
        "server": "vibe",
        "required": ["query"],
        "optional": [],
        "types": {"query": str},
        "description": "Execute a read-only SQL query against the AtlasTrinity PostgreSQL database for deep inspection",
    },
    "vibe_get_system_context": {
        "server": "vibe",
        "required": [],
        "optional": [],
        "description": "Retrieve current operational context (Session ID, latest tasks, last errors) for focused analysis",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MACOS-USE: GUI Automation
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_open_application_and_traverse": {
        "server": "macos-use",
        "required": ["identifier"],
        "optional": [],
        "types": {"identifier": str},
        "description": "Open/activate app and traverse accessibility tree",
    },
    "macos-use_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Click coordinates (x, y) and traverse tree",
    },
    "macos-use_type_and_traverse": {
        "server": "macos-use",
        "required": ["text"],
        "optional": ["pid", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "text": str,
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Type text and traverse tree",
    },
    "macos-use_press_key_and_traverse": {
        "server": "macos-use",
        "required": ["keyName"],
        "optional": ["pid", "modifierFlags", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "keyName": str,
            "modifierFlags": list,
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Press key (Return, Tab, Escape, etc.) with modifiers",
    },
    "macos-use_refresh_traversal": {
        "server": "macos-use",
        "required": [],
        "optional": ["pid"],
        "types": {"pid": int},
        "description": "Update accessibility tree traversal",
    },
    "macos-use_scroll_and_traverse": {
        "server": "macos-use",
        "required": ["direction"],
        "optional": ["pid", "amount", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "direction": str,
            "amount": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Scroll wheel action and traverse tree",
    },
    "macos-use_right_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Right-click and traverse tree",
    },
    "macos-use_double_click_and_traverse": {
        "server": "macos-use",
        "required": ["x", "y"],
        "optional": ["pid", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "x": (int, float),
            "y": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Double-click and traverse tree",
    },
    "macos-use_drag_and_drop_and_traverse": {
        "server": "macos-use",
        "required": ["startX", "startY", "endX", "endY"],
        "optional": ["pid", "showAnimation", "animationDuration", "traverseBefore", "traverseAfter"],
        "types": {
            "pid": int,
            "startX": (int, float),
            "startY": (int, float),
            "endX": (int, float),
            "endY": (int, float),
            "showAnimation": bool,
            "animationDuration": (int, float),
            "traverseBefore": bool,
            "traverseAfter": bool,
        },
        "description": "Drag and drop action and traverse tree",
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
        "description": "Manage windows: move, resize, minimize, maximize, make_front",
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
    # MACOS-USE: Productivity & Content
    # ─────────────────────────────────────────────────────────────────────────
    "macos-use_get_time": {
        "server": "macos-use",
        "required": [],
        "optional": ["timezone"],
        "types": {"timezone": str},
        "description": "Returns current system time",
    },
    "macos-use_fetch_url": {
        "server": "macos-use",
        "required": ["url"],
        "optional": [],
        "types": {"url": str},
        "description": "Fetch URL and convert HTML to markdown/text",
    },
    "macos-use_list_tools_dynamic": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "description": "List all available native macOS tools and schemas",
    },
    "macos-use_notes_create_note": {
        "server": "macos-use",
        "required": ["body"],
        "optional": ["folder"],
        "types": {"body": str, "folder": str},
        "description": "Create new note in Apple Notes",
    },
    "macos-use_notes_list_folders": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "description": "List all folders in Apple Notes",
    },
    "macos-use_notes_get_content": {
        "server": "macos-use",
        "required": ["name"],
        "optional": [],
        "types": {"name": str},
        "description": "Get HTML body of note by name",
    },
    "macos-use_mail_send": {
        "server": "macos-use",
        "required": ["to", "subject", "body"],
        "optional": [],
        "types": {"to": str, "subject": str, "body": str},
        "description": "Send email via Apple Mail",
    },
    "macos-use_mail_read_inbox": {
        "server": "macos-use",
        "required": [],
        "optional": ["limit"],
        "types": {"limit": int},
        "description": "Read recent message subjects from inbox",
    },
    "macos-use_finder_list_files": {
        "server": "macos-use",
        "required": [],
        "optional": ["path"],
        "types": {"path": str},
        "description": "List files in frontmost Finder window or path",
    },
    "macos-use_finder_get_selection": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "description": "Get POSIX paths of currently selected items in Finder",
    },
    "macos-use_finder_open_path": {
        "server": "macos-use",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Open POSIX path in Finder",
    },
    "macos-use_finder_move_to_trash": {
        "server": "macos-use",
        "required": ["path"],
        "optional": [],
        "types": {"path": str},
        "description": "Move item to Trash via Finder",
    },
    "macos-use_calendar_events": {
        "server": "macos-use",
        "required": ["start", "end"],
        "optional": [],
        "types": {"start": str, "end": str},
        "description": "Fetch calendar events for date range (ISO8601)",
    },
    "macos-use_create_event": {
        "server": "macos-use",
        "required": ["title", "date"],
        "optional": [],
        "types": {"title": str, "date": str},
        "description": "Create new calendar event",
    },
    "macos-use_reminders": {
        "server": "macos-use",
        "required": [],
        "optional": ["list"],
        "types": {"list": str},
        "description": "Fetch incomplete reminders",
    },
    "macos-use_create_reminder": {
        "server": "macos-use",
        "required": ["title"],
        "optional": [],
        "types": {"title": str},
        "description": "Create new reminder",
    },
    "macos-use_spotlight_search": {
        "server": "macos-use",
        "required": ["query"],
        "optional": [],
        "types": {"query": str},
        "description": "File search using Spotlight (mdfind)",
    },
    "macos-use_send_notification": {
        "server": "macos-use",
        "required": ["title", "message"],
        "optional": [],
        "types": {"title": str, "message": str},
        "description": "Send native macOS notification",
    },
    "macos-use_system_control": {
        "server": "macos-use",
        "required": ["action"],
        "optional": [],
        "types": {"action": str},
        "description": "System control (media, volume, brightness)",
    },
    "macos-use_get_clipboard": {
        "server": "macos-use",
        "required": [],
        "optional": [],
        "description": "Get current clipboard text",
    },
    "macos-use_set_clipboard": {
        "server": "macos-use",
        "required": ["text"],
        "optional": ["showAnimation", "animationDuration"],
        "types": {"text": str, "showAnimation": bool, "animationDuration": (int, float)},
        "description": "Set clipboard text content",
    },

    # ─────────────────────────────────────────────────────────────────────────
    # KNOWLEDGE & MEMORY
    # ─────────────────────────────────────────────────────────────────────────
    "create_entities": {
        "server": "memory",
        "required": ["entities"],
        "optional": [],
        "types": {"entities": list},
        "description": "Batch create or update entities in knowledge graph",
    },
    "add_observations": {
        "server": "memory",
        "required": ["name", "observations"],
        "optional": [],
        "types": {"name": str, "observations": list},
        "description": "Add observations to existing entity",
    },
    "get_entity": {
        "server": "memory",
        "required": ["name"],
        "optional": [],
        "types": {"name": str},
        "description": "Retrieve full entity details",
    },
    "list_entities": {
        "server": "memory",
        "required": [],
        "optional": [],
        "description": "List all entity names",
    },
    "search": {
        "server": "memory",
        "required": ["query"],
        "optional": ["limit"],
        "types": {"query": str, "limit": int},
        "description": "Semantic search in knowledge graph (returns closest entities)",
    },
    "create_relation": {
        "server": "memory",
        "required": ["source", "target", "relation"],
        "optional": [],
        "types": {"source": str, "target": str, "relation": str},
        "description": "Link two entities in the knowledge graph with a specific relationship",
    },
    "delete_entity": {
        "server": "memory",
        "required": ["name"],
        "optional": [],
        "types": {"name": str},
        "description": "Delete entity from graph",
    },
    "get_graph_json": {
        "server": "graph",
        "required": [],
        "optional": [],
        "description": "Export graph as JSON",
    },
    "generate_mermaid": {
        "server": "graph",
        "required": [],
        "optional": ["node_type"],
        "types": {"node_type": str},
        "description": "Generate a Mermaid.js flowchart of the graph (optional filter by node type)",
    },
    "get_node_details": {
        "server": "graph",
        "required": ["node_id"],
        "optional": [],
        "types": {"node_id": str},
        "description": "Retrieve full attributes and metadata for a specific graph node",
    },
    "get_related_nodes": {
        "server": "graph",
        "required": ["node_id"],
        "optional": [],
        "types": {"node_id": str},
        "description": "Find all nodes connected to a specific ID in the graph",
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
     - timeout_s: Timeout in seconds (default 3600)
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
   - BAD: "Running ls command in ~/Downloads"
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
    if any(x in task_lower for x in ["memory", "recall", "remember", "fact", "knowledge", "observation"]):
        return ["memory"]
    if any(x in task_lower for x in ["graph", "visualize", "diagram", "mermaid", "map", "relationship"]):
        return ["graph", "memory"]

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
