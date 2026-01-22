import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .config import CONFIG_ROOT, PROJECT_ROOT
from .config_loader import config
from .logger import logger
from .mcp_registry import get_server_for_tool, get_tool_schema
from .state_manager import state_manager


class ToolDispatcher:
    """
    Centralized dispatcher for MCP tools.
    Unifies tool name resolution, synonym mapping, and argument normalization.
    """

    # --- SYNONYMS & INTENT MAPPINGS ---
    TERMINAL_SYNONYMS = [
        "terminal",
        "bash",
        "zsh",
        "sh",
        "python",
        "python3",
        "pip",
        "pip3",
        "cmd",
        "run",
        "execute",
        "execute_command",
        "terminal_execute",
        "execute_terminal",
        "terminal.execute",
        "osascript",
        "applescript",
        "curl",
        "wget",
        "jq",
        "grep",
        "git",
        "npm",
        "npx",
        "brew",
        "mkdir",
        "ls",
        "cat",
        "rm",
        "mv",
        "cp",
        "touch",
        "sudo",
    ]

    FILESYSTEM_SYNONYMS = [
        "filesystem",
        "fs",
        "file",
        "files",
        "editor",
        "directory_tree",
        "list_directory",
        "read_file",
        "write_file",
        "tree",
    ]

    SERACH_SYNONYMS = []  # Deprecated

    VIBE_SYNONYMS = [
        "vibe",
        "vibe_prompt",
        "vibe_ask",
        "vibe_analyze_error",
        "vibe_smart_plan",
        "vibe_code_review",
        "vibe_implement_feature",
        "vibe_execute_subcommand",
        "vibe_list_sessions",
        "vibe_session_details",
        "vibe_which",
        "vibe_get_config",
        "vibe_configure_model",
        "vibe_set_mode",
        "vibe_configure_provider",
        "vibe_session_resume",
        "vibe_reload_config",
        "debug",
        "fix",
        "implement",
        "feature",
        "review",
        "plan",
        "ask",
        "question",
        "config",
        "model",
        "provider",
        "resume",
        "reload",
        "mode",
    ]

    BROWSER_SYNONYMS = [
        "browser",
        "puppeteer",
        "navigate",
        "google",
        "bing",
        "web",
        "web_search",
        "internet_search",
        "online_search",
    ]

    DUCKDUCKGO_SYNONYMS = [
        "duckduckgo",
        "ddg",
        "duckduckgo-search",
        "duckduckgo_search",
        "search_web",
        "web_search",
    ]

    KNOWLEDGE_SYNONYMS = [
        "memory",
        "knowledge",
        "entity",
        "entities",
        "observation",
        "observations",
        "fact",
        "recall",
        "remember",
        "store_fact",
        "add_memory",
        "relationship",
        "relation",
    ]

    GRAPH_SYNONYMS = [
        "graph",
        "visualization",
        "diagram",
        "mermaid",
        "flowchart",
        "nodes",
        "edges",
        "node_details",
        "related_nodes",
        "traverse",
    ]

    REDIS_SYNONYMS = [
        "redis",
        "cache",
        "state_inspection",
        "session_storage",
        "flags",
        "retry_pending",
        "restart_pending",
    ]

    DEVTOOLS_SYNONYMS = [
        "devtools",
        "lint",
        "linter",
        "check",
        "inspect",
        "inspector",
        "validate",
        "health",
        "ruff",
        "oxlint",
        "knip",
        "pyrefly",
    ]

    CONTEXT7_SYNONYMS = [
        "context7",
        "c7",
        "docs",
        "documentation",
        "library",
        "library_search",
        "api_docs",
        "lookup",
    ]

    GITHUB_SYNONYMS = [
        "github",
        "repo",
        "repository",
        "pull_request",
        "pr",
        "issue",
        "issues",
        "gh",
        "git_hub",
    ]

    MACOS_MAP = {
        "click": "macos-use_click_and_traverse",
        "type": "macos-use_type_and_traverse",
        "write": "macos-use_type_and_traverse",
        "press": "macos-use_press_key_and_traverse",
        "hotkey": "macos-use_press_key_and_traverse",
        "refresh": "macos-use_refresh_traversal",
        "screenshot": "macos-use_take_screenshot",
        "vision": "macos-use_analyze_screen",
        "ocr": "macos-use_analyze_screen",
        "open": "macos-use_open_application_and_traverse",
        "launch": "macos-use_open_application_and_traverse",
        "scroll": "macos-use_scroll_and_traverse",
        "fetch": "macos-use_fetch_url",
        "fetch_url": "macos-use_fetch_url",
        "time": "macos-use_get_time",
        "get_time": "macos-use_get_time",
        "notification": "macos-use_send_notification",
        "run_applescript": "macos-use_run_applescript",
        "applescript": "macos-use_run_applescript",
        "spotlight": "macos-use_spotlight_search",
        "spotlight_search": "macos-use_spotlight_search",
        "clipboard_set": "macos-use_set_clipboard",
        "set_clipboard": "macos-use_set_clipboard",
        "clipboard_get": "macos-use_get_clipboard",
        "get_clipboard": "macos-use_get_clipboard",
        # Notes tools
        "create_note": "macos-use_notes_create_note",
        "notes_create": "macos-use_notes_create_note",
        "list_notes": "macos-use_notes_list_folders",
        "notes_list": "macos-use_notes_list_folders",
        "get_note": "macos-use_notes_get_content",
        "read_note": "macos-use_notes_get_content",
        "search_notes": "macos-use_notes_get_content",
        "notes_get": "macos-use_notes_get_content",
        # Finder tools
        "finder_open": "macos-use_finder_open_path",
        "open_path": "macos-use_finder_open_path",
        "list_files": "macos-use_finder_list_files",
        "finder_list": "macos-use_finder_list_files",
        "finder_selection": "macos-use_finder_get_selection",
        "get_selection": "macos-use_finder_get_selection",
        "trash": "macos-use_finder_move_to_trash",
        "move_to_trash": "macos-use_finder_move_to_trash",
        # Calendar/Reminders
        "calendar_events": "macos-use_calendar_events",
        "create_event": "macos-use_create_event",
        "reminders": "macos-use_reminders",
        "create_reminder": "macos-use_create_reminder",
        # Mail
        "send_mail": "macos-use_mail_send",
        "mail_send": "macos-use_mail_send",
        "read_inbox": "macos-use_mail_read_inbox",
        "mail_read": "macos-use_mail_read_inbox",
        # Media/System
        "system_control": "macos-use_system_control",
        "media": "macos-use_system_control",
        # Explicit Terminal support within macos-use routing
        "terminal": "execute_command",
        "execute_command": "execute_command",
        "shell": "execute_command",
        "bash": "execute_command",
        "zsh": "execute_command",
        "sh": "execute_command",
        "ls": "macos-use_finder_list_files",
        "cd": "execute_command",
        "pwd": "execute_command",
        "echo": "execute_command",
        "cat": "execute_command",
        "grep": "execute_command",
        "curl": "macos-use_fetch_url",
        "wget": "macos-use_fetch_url",
        "date": "macos-use_get_time",
        "notify": "macos-use_send_notification",
        "alert": "macos-use_send_notification",
        "find": "macos-use_spotlight_search",
        "mdfind": "macos-use_spotlight_search",
        "right_click": "macos-use_right_click_and_traverse",
        "double_click": "macos-use_double_click_and_traverse",
        "drag": "macos-use_drag_and_drop_and_traverse",
        "drop": "macos-use_drag_and_drop_and_traverse",
        "drag_and_drop": "macos-use_drag_and_drop_and_traverse",
        # Discovery
        "list_tools": "macos-use_list_tools_dynamic",
        "discovery": "macos-use_list_tools_dynamic",
    }

    MACOS_USE_PRIORITY = {
        "bash",
        "zsh",
        "sh",
        "execute",
        "run",
        "cmd",
        "command",
        "git",
        "npm",
        "npx",
        "pip",
        "brew",
        "curl",
        "wget",
        "time",
        "clock",
        "date",
        "fetch",
        "url",
        "scrape",
        "volume",
        "brightness",
        "mute",
        "play",
        "pause",
        "calendar",
        "event",
        "reminder",
        "note",
        "mail",
        "email",
        "finder",
        "trash",
        "spotlight",
        "applescript",
        "osascript",
    }

    def __init__(self, mcp_manager):
        self.mcp_manager = mcp_manager
        self._tasks = set()  # To prevent GC of long-running tasks
        self._current_pid: int | None = None
        self._total_calls = 0
        self._macos_use_calls = 0

    def set_pid(self, pid: int | None):
        """Update the currently tracked PID for macOS automation."""
        self._current_pid = pid

    # Common hallucinated tool names that LLMs generate but don't exist
    HALLUCINATED_TOOLS = {
        "evaluate": "No 'evaluate' tool exists. Use vibe_code_review for code evaluation or execute_command for running tests.",
        "assess": "No 'assess' tool exists. Use vibe_code_review for assessment.",
        "verify": "No 'verify' tool exists. Use execute_command to run verification commands.",
        "validate": "No 'validate' tool exists. Use execute_command to run validation scripts.",
        "check": "No 'check' tool exists. Use execute_command for running check commands.",
        "test": "No 'test' tool exists. Use execute_command('npm test') or similar.",
        "compile": "No 'compile' tool exists. Use execute_command with appropriate build command.",
        "build": "No 'build' tool exists. Use execute_command('npm run build') or similar.",
        "deploy": "No 'deploy' tool exists. Use execute_command with deployment scripts.",
        "run": "Use execute_command for running arbitrary commands.",
    }

    async def resolve_and_dispatch(
        self, tool_name: str | None, args: dict[str, Any], explicit_server: str | None = None
    ) -> dict[str, Any]:
        """
        The main entry point for dispatching a tool call.
        Resolves the tool name, normalizes arguments, and executes the call via MCPManager.
        """
        try:
            # 1. Basic cleaning and normalization
            tool_name = (tool_name or "").strip().lower()
            if not isinstance(args, dict):
                args = {}

            # 2. Check for known hallucinated tools first
            if tool_name in self.HALLUCINATED_TOOLS:
                suggestion = self.HALLUCINATED_TOOLS[tool_name]
                logger.warning(
                    f"[DISPATCHER] Hallucinated tool detected: '{tool_name}'. {suggestion}"
                )
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' does not exist. {suggestion}",
                    "hallucinated": True,
                }

            # 3. Heuristic inference if tool_name is missing
            if not tool_name:
                tool_name = self._infer_tool_from_args(args)

            # 4. Handle Dot or Underscore Notation (server.tool or server_tool)
            if "." in tool_name:
                parts = tool_name.split(".", 1)
                explicit_server = parts[0]
                tool_name = parts[1]
            else:
                from .mcp_registry import SERVER_CATALOG, TOOL_SCHEMAS

                # If tool_name is ALREADY a canonical tool in the registry, leave it alone
                # (e.g., 'redis_get' or 'macos-use_fetch_url')
                if tool_name not in TOOL_SCHEMAS:
                    # Heuristic: check if tool_name starts with a known server name
                    # Sort by length descending to catch 'duckduckgo-search' before 'duckduckgo'
                    sorted_servers = sorted(SERVER_CATALOG.keys(), key=len, reverse=True)
                    for s_name in sorted_servers:
                        # Check for server_tool or server__tool
                        # Also handle hyphenated servers appearing as underscores (e.g. duckduckgo_search_...)
                        prefixes = [f"{s_name}_", f"{s_name.replace('-', '_')}_"]
                        for prefix in prefixes:
                            if tool_name.startswith(prefix):
                                potential_tool = tool_name[len(prefix) :]
                                if potential_tool.startswith("_"):
                                    potential_tool = potential_tool[1:]

                                # Only strip if the remaining part exists as a tool or is recognized
                                explicit_server = s_name
                                tool_name = potential_tool
                                logger.info(
                                    f"[DISPATCHER] Normalized {prefix}{tool_name} to {explicit_server}.{tool_name}"
                                )
                                break
                        if explicit_server:
                            break

            # 5. Intelligent Routing with macOS-use Priority
            server, resolved_tool, normalized_args = self._intelligent_routing(
                tool_name, args, explicit_server
            )

            # Special case: System tools handled internally
            if server in {"_trinity_native", "system"}:
                return await self._handle_system(resolved_tool, normalized_args)

            if not server:
                # Provide helpful suggestions for unknown tools
                from .mcp_registry import get_all_tool_names

                all_tools = get_all_tool_names()

                # Find similar tool names (simple substring match)
                similar = [
                    t for t in all_tools if tool_name in t.lower() or t.lower() in tool_name
                ][:5]
                suggestion = f" Did you mean: {', '.join(similar)}" if similar else ""

                logger.warning(f"[DISPATCHER] Unknown tool: '{tool_name}'.{suggestion}")
                return {
                    "success": False,
                    "error": f"Could not resolve server for tool: '{tool_name}'.{suggestion}",
                    "unknown_tool": True,
                }

            # 6. Validate and normalize arguments before calling MCP
            validated_args = self._validate_args(resolved_tool, normalized_args)
            if validated_args.get("__validation_error__"):
                error_msg = validated_args.pop("__validation_error__")
                logger.warning(
                    f"[DISPATCHER] Argument validation failed for {resolved_tool}: {error_msg}"
                )
                return {
                    "success": False,
                    "error": f"Invalid arguments for '{resolved_tool}': {error_msg}",
                    "validation_error": True,
                }

            # 7. Track metrics
            self._total_calls += 1
            if server == "macos-use":
                self._macos_use_calls += 1

            # 8. Metrics & Final Dispatch via MCPManager
            logger.info(
                f"[DISPATCHER] Calling {server}.{resolved_tool} with {list(validated_args.keys())}"
            )

            result = await self.mcp_manager.call_tool(server, resolved_tool, validated_args)

            # 9. Check for "Tool not found" errors from MCP and provide guidance
            if isinstance(result, dict) and result.get("error"):
                error_msg = str(result.get("error", ""))
                if "not found" in error_msg.lower() or "-32602" in error_msg:
                    logger.warning(
                        f"[DISPATCHER] Tool not found on server: {server}.{resolved_tool}"
                    )
                    result["suggestion"] = (
                        f"Tool '{resolved_tool}' may not exist on server '{server}'. Check available tools with list_tools."
                    )

            return result

        except Exception as e:
            logger.error(f"[DISPATCHER] Dispatch failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _validate_args(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and normalize arguments according to tool schema.
        Returns args with __validation_error__ key if validation failed.
        """
        from .mcp_registry import get_tool_schema

        schema = get_tool_schema(tool_name)
        if not schema:
            # No schema found - pass through without validation
            return args if isinstance(args, dict) else {}

        validated = dict(args) if isinstance(args, dict) else {}

        # Check required arguments
        required = schema.get("required", [])
        missing = [r for r in required if r not in validated or validated[r] is None]
        if missing:
            validated["__validation_error__"] = f"Missing required arguments: {', '.join(missing)}"
            return validated

        # Type conversion
        types_map = schema.get("types", {})
        for key, expected_type in types_map.items():
            if key in validated and validated[key] is not None:
                try:
                    value = validated[key]
                    if expected_type == "str" and not isinstance(value, str):
                        validated[key] = str(value)
                    elif expected_type == "int" and not isinstance(value, int):
                        validated[key] = int(float(value))
                    elif expected_type == "float" and not isinstance(value, int | float):
                        validated[key] = float(value)
                    elif expected_type == "bool" and not isinstance(value, bool):
                        validated[key] = str(value).lower() in ("true", "1", "yes")
                    elif expected_type == "list" and not isinstance(value, list):
                        if isinstance(value, str):
                            # Try to parse as JSON list
                            import json

                            try:
                                parsed = json.loads(value)
                                if isinstance(parsed, list):
                                    validated[key] = parsed
                                else:
                                    validated[key] = [value]
                            except json.JSONDecodeError:
                                validated[key] = [value]
                        else:
                            validated[key] = [value]
                except (ValueError, TypeError) as e:
                    logger.warning(f"[DISPATCHER] Type conversion failed for {key}: {e}")

        return validated

    def _infer_tool_from_args(self, args: dict[str, Any]) -> str:
        """Infers tool name from common argument patterns when missing."""
        action = str(args.get("action", "")).lower()
        command = str(args.get("command", args.get("cmd", ""))).lower()
        path = str(args.get("path", "")).lower()

        if "vibe" in action or "vibe" in command:
            return "vibe"
        if any(kw in action for kw in ["click", "type", "press", "screenshot", "scroll"]):
            return "macos-use"
        if any(kw in action for kw in ["read", "write", "list", "save", "delete"]) or path:
            return "filesystem"
        if any(kw in action for kw in ["browser", "puppeteer", "navigate", "google", "search"]):
            return "puppeteer"
        if command:
            return "terminal"

        return action or "terminal"

    def _can_macos_use_handle(self, tool_name: str) -> bool:
        """Check if macOS-use can handle this tool based on priority set."""
        tool_lower = tool_name.lower()

        # Direct check in priority set
        if any(priority in tool_lower for priority in self.MACOS_USE_PRIORITY):
            return True

        # Check if it's already a macos-use tool
        if tool_lower.startswith("macos-use") or tool_lower.startswith("macos_use_"):
            return True

        # Check MACOS_MAP
        if tool_lower in self.MACOS_MAP:
            return True

        return False

    def _intelligent_routing(
        self, tool_name: str, args: dict[str, Any], explicit_server: str | None = None
    ) -> tuple[str | None, str, dict[str, Any]]:
        """
        Intelligent tier-based routing with macOS-use priority.
        Now delegates to BehaviorEngine for config-driven routing.
        """
        from .behavior_engine import behavior_engine

        # Fix for generic 'memory' tool call (LLM hallucination)
        if tool_name.lower() == "memory" and "query" in args:
            return "memory", "search", args

        # Delegate to behavior engine for routing (replaces 150+ lines of hardcoded logic)
        try:
            server, resolved_tool, normalized_args = behavior_engine.route_tool(
                tool_name, args, explicit_server
            )

            if server:
                logger.debug(
                    f"[DISPATCHER] BehaviorEngine routing: {tool_name} -> {server}.{resolved_tool}"
                )
                return server, resolved_tool, normalized_args
        except Exception as e:
            logger.warning(
                f"[DISPATCHER] BehaviorEngine routing failed: {e}, falling back to registry"
            )

        # Fallback: Use registry-based resolution
        return self._resolve_tool_and_args(tool_name, args, explicit_server)

    def get_coverage_stats(self) -> dict[str, Any]:
        """Get macOS-use coverage statistics."""
        coverage_pct = (
            (self._macos_use_calls / self._total_calls * 100) if self._total_calls > 0 else 0
        )
        return {
            "total_calls": self._total_calls,
            "macos_use_calls": self._macos_use_calls,
            "coverage_percentage": round(coverage_pct, 2),
            "target": 90.0,
        }

    def _resolve_tool_and_args(
        self, tool_name: str, args: dict[str, Any], explicit_server: str | None = None
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Resolves tool name to canonical form and normalizes arguments."""

        # --- TERMINAL ROUTING ---
        if tool_name in self.TERMINAL_SYNONYMS or explicit_server == "terminal":
            return self._handle_terminal(tool_name, args)

        # --- FILESYSTEM ROUTING ---
        if tool_name in self.FILESYSTEM_SYNONYMS or explicit_server == "filesystem":
            return self._handle_filesystem(tool_name, args)

        # --- BROWSER ROUTING ---
        # Exclude 'search' from browser routing to ensure it goes to memory server as per tool schemas
        if (
            (tool_name in self.BROWSER_SYNONYMS and tool_name != "search")
            or any(tool_name.startswith(x) for x in ["puppeteer_", "browser_"])
            or explicit_server == "puppeteer"
        ):
            return self._handle_browser(tool_name, args)

        # --- VIBE ROUTING ---
        if tool_name in self.VIBE_SYNONYMS or explicit_server == "vibe":
            return self._handle_vibe(tool_name, args)

        # --- NOTES ROUTING (Redirect to macos-use) ---
        if explicit_server == "notes" or any(tool_name.startswith(p) for p in ["notes_", "note_"]):
            return self._handle_macos_use(tool_name, args)

        # --- MACOS-USE ROUTING ---
        if (
            tool_name.startswith("macos-use")
            or tool_name.startswith("macos_use_")
            or tool_name in self.MACOS_MAP
            or explicit_server == "macos-use"
        ):
            return self._handle_macos_use(tool_name, args)

        # --- SEQUENTIAL THINKING ---
        if tool_name in ["sequential-thinking", "sequentialthinking", "think"]:
            return "sequential-thinking", "sequentialthinking", args

        # --- DEVTOOLS ROUTING ---
        if tool_name in self.DEVTOOLS_SYNONYMS or explicit_server == "devtools":
            # Map generic terms to specific tools
            if tool_name in ["lint", "linter", "ruff"]:
                return "devtools", "devtools_lint_python", args
            if tool_name in ["oxlint", "js_lint"]:
                return "devtools", "devtools_lint_js", args
            if tool_name in ["inspect", "inspector"]:
                return "devtools", "devtools_launch_inspector", args
            if tool_name in ["health", "check"]:
                # Check 'check_code' vs 'health_check'
                if "mcp" in str(args):
                    return "devtools", "devtools_check_mcp_health", args
                # Default generic 'check' might be ambiguous, but let's assume health if mcp mentioned
                return "devtools", "devtools_check_mcp_health", args

            return "devtools", tool_name, args

        # --- CONTEXT7 ROUTING ---
        if tool_name in self.CONTEXT7_SYNONYMS or explicit_server == "context7":
            # Map generic "docs" or "search" to c7_search
            if tool_name in ["docs", "documentation", "lookup", "library"]:
                return "context7", "c7_search", args
            return "context7", tool_name, args

        # --- GIT LEGACY ROUTING ---
        if tool_name.startswith("git_") or explicit_server == "git":
            return self._handle_legacy_git(tool_name, args)

        # --- FALLBACK: USE REGISTRY ---
        # Normalize hyphenated tool names to underscores for Python-based servers
        if tool_name == "duckduckgo-search":
            tool_name = "duckduckgo_search"
        if tool_name == "whisper-stt":
            tool_name = "transcribe_audio"

        server = explicit_server or get_server_for_tool(tool_name)
        if not server:
            # Try registry-based name mapping (e.g. 'read_file' -> 'filesystem' server)
            schema = get_tool_schema(tool_name)
            if schema:
                server = schema.get("server")

        return server, tool_name, args

    def _handle_legacy_git(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """Maps legacy git_server tools to macos-use execute_command."""
        subcommand = tool_name.replace("git_", "").replace("_", "-")  # git_status -> status

        # Base command
        cmd_parts = ["git", subcommand]

        # Heuristic argument mapping
        if "path" in args:  # implicit cwd usually, but for git command usually we run IN that dir
            # We rely on mcp_manager to handle 'cwd' via chaining or just assume '.' is target
            pass

        # Simple flags mapping
        if args.get("porcelain"):
            cmd_parts.append("--porcelain")
        if args.get("staged"):
            cmd_parts.append("--staged")
        if args.get("message"):
            cmd_parts.extend(["-m", f'"{args["message"]}"'])
        if args.get("branch"):
            cmd_parts.append(args["branch"])
        if args.get("target"):
            cmd_parts.append(args["target"])

        # Construct command
        full_command = " ".join(cmd_parts)

        new_args = {"command": full_command}
        if "path" in args:
            # git_server used 'path' as cwd.
            # execute_command doesn't natively support cwd param in mcp-server-macos-use (it runs in user home usually)
            # So we chain it: cd path && git ...
            path = args["path"]
            new_args["command"] = f"cd {path} && {full_command}"

        return "macos-use", "execute_command", new_args

    def _handle_terminal(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """Standardizes terminal command execution via macos-use."""
        cmd = (
            args.get("command")
            or args.get("cmd")
            or args.get("code")
            or args.get("script")
            or args.get("args")
            or args.get("action")
        )

        # Handle cases where tool_name IS the command (mkfs, ls, etc)
        if tool_name in [
            "mkdir",
            "ls",
            "cat",
            "rm",
            "mv",
            "cp",
            "touch",
            "sudo",
            "git",
            "npm",
            "npx",
            "brew",
        ]:
            if isinstance(cmd, str):
                cmd = f"{tool_name} {cmd}".strip()
            else:
                cmd = tool_name

        args["command"] = str(cmd) if cmd else ""
        return "macos-use", "execute_command", args

    def _handle_filesystem(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """Maps filesystem synonyms to canonical tools."""
        action = args.get("action") or tool_name

        # Action mappings
        mapping = {
            "list_dir": "list_directory",
            "ls": "list_directory",
            "mkdir": "create_directory",
            "write": "write_file",
            "save": "write_file",
            "read": "read_file",
            "cat": "read_file",
            "exists": "get_file_info",
            "directory_tree": "list_directory",
            "tree": "list_directory",
        }
        resolved_tool = mapping.get(action, action)

        # If the tool name itself was the action
        if resolved_tool == "filesystem":
            resolved_tool = "read_file"  # Default

        return "filesystem", resolved_tool, args

    def _handle_browser(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """Maps browser synonyms to Puppeteer tools.

        IMPORTANT: 'search' must NEVER be routed through this method.
        Search functionality is handled exclusively by the memory server.
        This ensures search results are properly stored and accessible for knowledge graph operations.
        """
        action = args.get("action") or tool_name

        # Critical safeguard: prevent 'search' from being routed to puppeteer directly
        # Instead of crashing, we route to duckduckgo for actual web search
        if tool_name == "search" or action == "search":
            logger.info("[DISPATCHER] Redirecting browser search to duckduckgo-search")
            return "duckduckgo-search", "duckduckgo_search", args

        mapping = {
            "google": "puppeteer_navigate",
            "bing": "puppeteer_navigate",
            "navigate": "puppeteer_navigate",
            "browse": "puppeteer_navigate",
            "web_search": "puppeteer_navigate",
            "internet_search": "puppeteer_navigate",
            "online_search": "puppeteer_navigate",
            "screenshot": "puppeteer_screenshot",
            "click": "puppeteer_click",
            "type": "puppeteer_fill",
            "fill": "puppeteer_fill",
        }

        resolved_tool = mapping.get(action, action)

        # Ensure 'puppeteer_' prefix if not already present
        if not resolved_tool.startswith("puppeteer_") and resolved_tool != "puppeteer":
            resolved_tool = f"puppeteer_{resolved_tool}"

        # If it was just 'browser' or 'puppeteer'
        if resolved_tool in ["browser", "puppeteer", "puppeteer_browser", "puppeteer_puppeteer"]:
            resolved_tool = "puppeteer_navigate"

        return "puppeteer", resolved_tool, args

    def _handle_vibe(self, tool_name: str, args: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Normalizes Vibe AI tool calls and arguments."""
        # Tool name normalization
        vibe_map = {
            "vibe": "vibe_prompt",
            "prompt": "vibe_prompt",
            "vibe_prompt": "vibe_prompt",
            "ask": "vibe_ask",
            "vibe_ask": "vibe_ask",
            "question": "vibe_ask",
            "plan": "vibe_smart_plan",
            "smart_plan": "vibe_smart_plan",
            "vibe_smart_plan": "vibe_smart_plan",
            "debug": "vibe_analyze_error",
            "fix": "vibe_analyze_error",
            "vibe_analyze_error": "vibe_analyze_error",
            "analyze_error": "vibe_analyze_error",
            "review": "vibe_code_review",
            "vibe_code_review": "vibe_code_review",
            "code_review": "vibe_code_review",
            "implement": "vibe_implement_feature",
            "feature": "vibe_implement_feature",
            "vibe_implement_feature": "vibe_implement_feature",
            "implement_feature": "vibe_implement_feature",
            "subcommand": "vibe_execute_subcommand",
            "vibe_execute_subcommand": "vibe_execute_subcommand",
            "sessions": "vibe_list_sessions",
            "vibe_list_sessions": "vibe_list_sessions",
            "session_details": "vibe_session_details",
            "vibe_session_details": "vibe_session_details",
            "which": "vibe_which",
            "vibe_which": "vibe_which",
            "config": "vibe_get_config",
            "get_config": "vibe_get_config",
            "vibe_get_config": "vibe_get_config",
            "model": "vibe_configure_model",
            "configure_model": "vibe_configure_model",
            "switch_model": "vibe_configure_model",
            "vibe_configure_model": "vibe_configure_model",
            "mode": "vibe_set_mode",
            "set_mode": "vibe_set_mode",
            "vibe_set_mode": "vibe_set_mode",
            "provider": "vibe_configure_provider",
            "configure_provider": "vibe_configure_provider",
            "vibe_configure_provider": "vibe_configure_provider",
            "resume": "vibe_session_resume",
            "continue": "vibe_session_resume",
            "vibe_session_resume": "vibe_session_resume",
            "reload": "vibe_reload_config",
            "reload_config": "vibe_reload_config",
            "vibe_reload_config": "vibe_reload_config",
        }
        resolved_tool = vibe_map.get(tool_name, tool_name)
        if not resolved_tool.startswith("vibe_"):
            resolved_tool = f"vibe_{resolved_tool}"

        # Argument normalization
        if "prompt" not in args:
            if "objective" in args:
                args["prompt"] = args["objective"]
            elif "question" in args:
                args["prompt"] = args["question"]
            elif "error_message" in args:
                args["prompt"] = args["error_message"]

        # Enforce defaults/timeouts - Vibe tasks can be long-running
        if "timeout_s" not in args:
            # Try mcp.vibe config first, then default to 1 hour (3600s)
            vibe_cfg = config.get("mcp", {}).get("vibe", {})
            args["timeout_s"] = float(vibe_cfg.get("timeout_s", 3600))

        # Enforce absolute CWD or workspace from config
        if not args.get("cwd"):
            system_config = config.get("system", {})
            # Recommended default is CONFIG_ROOT / "workspace" if not in config
            workspace_str = system_config.get("workspace_path", str(CONFIG_ROOT / "workspace"))
            workspace = Path(workspace_str).expanduser().absolute()
            args["cwd"] = str(workspace)
            workspace.mkdir(parents=True, exist_ok=True)

        # Verify repository path for self-healing
        system_config = config.get("system", {})
        repo_path = Path(system_config.get("repository_path", PROJECT_ROOT)).expanduser().absolute()
        if (repo_path / ".git").exists():
            logger.info(f"[DISPATCHER] Repository root verified for self-healing: {repo_path}")
        else:
            logger.warning(
                f"[DISPATCHER] Repository root at {repo_path} is NOT a git repo. Self-healing might be limited."
            )

        return "vibe", resolved_tool, args

    async def _handle_system(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handles internal Trinity system tools."""
        if tool_name == "restart_mcp_server":
            server_to_restart = args.get("server_name")
            if not server_to_restart:
                return {"success": False, "error": "Missing 'server_name' argument."}

            logger.info(f"[SYSTEM] Restarting MCP server: {server_to_restart}")
            success = await self.mcp_manager.restart_server(server_to_restart)
            return {
                "success": success,
                "result": f"Server '{server_to_restart}' restart {'successful' if success else 'failed'}.",
            }

        elif tool_name == "query_db":
            # For now, we don't expose raw SQL to agents for safety, but we could implement specific queries
            return {
                "success": False,
                "error": "Direct DB queries via LLM are currently restricted for safety.",
            }

        elif tool_name == "restart_application":
            reason = args.get("reason", "Manual restart triggered")
            logger.warning(f"[SYSTEM] Application restart triggered: {reason}")

            # trigger async restart to allow this request to complete
            import os
            import sys

            async def delayed_restart():
                # Set restart_pending flag in Redis for resumption logic
                if state_manager and state_manager.available:
                    restart_metadata = {
                        "reason": reason,
                        "timestamp": datetime.now().isoformat(),
                        "session_id": "current",  # Or a specific active session ID if known
                    }
                    cast(Any, state_manager).redis.set(
                        cast(Any, state_manager)._key("restart_pending"),
                        json.dumps(restart_metadata),
                    )
                    logger.info("[SYSTEM] restart_pending flag set in Redis.")

                await asyncio.sleep(2.0)
                logger.info("[SYSTEM] Executing os.execv restart now...")
                os.execv(sys.executable, [sys.executable, *sys.argv])

            task = asyncio.create_task(delayed_restart())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

            return {
                "success": True,
                "result": "Initiating graceful restart sequence. I will be back in a moment.",
            }

        elif tool_name in {"system", "status"}:
            # Generic status/meta tool for informational steps
            return {
                "success": True,
                "result": args.get("message") or args.get("action") or "Operation noted by system.",
            }

        return {"success": False, "error": f"Unknown system tool: {tool_name}"}

    def _handle_macos_use(
        self, tool_name: str, args: dict[str, Any]
    ) -> tuple[str, str, dict[str, Any]]:
        """Standardizes macos-use GUI and productivity tool calls."""
        # Clean prefix if it exists
        clean_name = tool_name
        if tool_name.startswith("macos-use_"):
            clean_name = tool_name[10:]
        elif tool_name.startswith("macos_use_"):
            clean_name = tool_name[10:]
        elif tool_name.startswith("git_"):
            return self._handle_legacy_git(tool_name, args)
        elif tool_name == "macos-use":
            # Infer tool from arguments
            if "identifier" in args:
                clean_name = "open"
            elif "x" in args:
                clean_name = "click"
            elif "text" in args:
                clean_name = "type"
            elif "path" in args:
                clean_name = "finder_list"
            elif "url" in args:
                clean_name = "fetch"
            elif "command" in args:
                clean_name = "terminal"
            else:
                clean_name = "screenshot"

        resolved_tool = self.MACOS_MAP.get(clean_name, tool_name)

        # FINAL SAFETY: If we still have 'macos-use' as a method name, it's definitely an error.
        # Try one last heuristic based on args.
        if resolved_tool == "macos-use":
            if "command" in args or "cmd" in args:
                resolved_tool = "execute_command"
            elif "path" in args:
                resolved_tool = "macos-use_finder_open_path"
            else:
                resolved_tool = "macos-use_take_screenshot"
            logger.info(f"[DISPATCHER] Last-resort mapping macos-use -> {resolved_tool}")

        if resolved_tool == "macos-use_fetch_url":
            if "urls" in args and "url" not in args:
                urls = args.get("urls")
                if isinstance(urls, list) and len(urls) > 0:
                    args["url"] = urls[0]
                    logger.info(f"[DISPATCHER] Patched fetch: urls[0] -> url ({args['url']})")

        # Inject PID if missing
        if self._current_pid and "pid" not in args:
            args["pid"] = self._current_pid

        # Standardize 'identifier' for app opening
        if resolved_tool == "macos-use_open_application_and_traverse" and "identifier" not in args:
            args["identifier"] = args.get("app_name") or args.get("name") or args.get("app") or ""

        return "macos-use", resolved_tool, args
