import asyncio
import json
import os
import shutil
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _import_mcp_sdk():
    original_sys_path = list(sys.path)
    try:
        src_dir = str(Path(__file__).resolve().parents[1])
        sys.path = [p for p in sys.path if str(p) != src_dir]

        from mcp.client.session import ClientSession as _ClientSession  # noqa: E402
        from mcp.client.stdio import StdioServerParameters as _StdioServerParameters  # noqa: E402
        from mcp.client.stdio import stdio_client as _stdio_client  # noqa: E402
        from mcp.types import LoggingMessageNotification as _LoggingMessageNotification  # noqa: E402

        return _ClientSession, _StdioServerParameters, _stdio_client, _LoggingMessageNotification
    finally:
        sys.path = original_sys_path


# Import preflight utilities (uses npm under the hood for registry checks)
try:
    from .mcp_preflight import check_package_arg_for_tool  # noqa: E402
except Exception:
    # Fallback: if preflight not available, define a permissive stub
    def check_package_arg_for_tool(arg: str, tool_cmd: str = "npx") -> bool:  # type: ignore
        return True


try:
    ClientSession, StdioServerParameters, stdio_client, LoggingMessageNotification = _import_mcp_sdk()
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    LoggingMessageNotification = None  # type: ignore
from .config import MCP_DIR  # noqa: E402
from .config_loader import config  # noqa: E402
from .logger import logger  # noqa: E402
from .db.manager import db_manager  # noqa: E402
from sqlalchemy import text  # noqa: E402


class MCPManager:
    """
    Manages persistent connections to MCP servers.

    Note:
      - Previously we used a single shared AsyncExitStack which caused AnyIO
        "Attempted to exit cancel scope in a different task than it was entered in"
        errors when contexts were created in one task and closed in another.
      - To avoid that, each server runs in its own connection task which enters
        and exits the stdio client within the same task. This ensures anyio
        cancel scopes/task-groups are exited in the same task they were entered.
    """

    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        # Per-server connection tasks and control structures
        self._connection_tasks: Dict[str, asyncio.Task] = {}
        self._close_events: Dict[str, asyncio.Event] = {}
        self._session_futures: Dict[str, asyncio.Future] = {}
        self.config = self._load_config()
        self._lock = asyncio.Lock()
        self._log_callbacks = []
        
        # Unified tool dispatching
        from .tool_dispatcher import ToolDispatcher
        self.dispatcher = ToolDispatcher(self)
        
        # Controls for restart concurrency and retry/backoff
        # Limit number of concurrent restarts to avoid forking storms
        self._restart_semaphore = asyncio.Semaphore(4)
        self._max_restart_attempts = int(config.get("mcp_enhanced.max_restart_attempts", 5))
        self._restart_backoff_base = float(
            config.get("mcp_enhanced.restart_backoff_base", 0.5)
        )  # seconds

    def _load_config(self) -> Dict[str, Any]:
        """Load MCP config from the global user config folder."""
        try:
            config_path = MCP_DIR / "config.json"
            if config_path.exists():
                logger.info(f"Loading MCP config from: {config_path}")
                with open(config_path, "r", encoding="utf-8") as f:
                    raw_config = json.load(f)
                return self._process_config(raw_config)

            logger.warning(f"MCP Config not found at: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {}

    def _process_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter disabled servers and substitute environment variables"""
        processed = {
            "mcpServers": {
                "_defaults": {
                    "connect_timeout": float(config.get("mcp_enhanced.connection_timeout", 30))
                }
            }
        }

        import re  # noqa: E402

        def _substitute_placeholders(value: Any, missing: List[str]) -> Any:
            if not isinstance(value, str):
                return value

            def replace_match(match):
                var_name = match.group(1)
                if var_name == "PROJECT_ROOT":
                    from .config import PROJECT_ROOT  # noqa: E402
                    return str(PROJECT_ROOT)
                if var_name == "CONFIG_ROOT":
                    from .config import CONFIG_ROOT  # noqa: E402
                    return str(CONFIG_ROOT)
                if var_name == "HOME":
                    return str(Path.home())
                
                # Fallback to environment variables
                env_val = os.getenv(var_name)
                if env_val is None:
                    missing.append(var_name)
                    return match.group(0)
                return env_val

            result = re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace_match, value)

            # --- PRODUCTION PATH RESOLUTION ---
            # If we are in a packaged app (frozen), many files from vendor/ are moved to bin/
            # This logic tries to find the binary if the original path doesn't exist.
            if (
                getattr(sys, "frozen", False)
                or "Resources/app.asar" in result
                or "/Resources/brain" in result
            ):
                if not os.path.exists(result):
                    # Robust resolution for packaged binary
                    from .config import PROJECT_ROOT  # noqa: E402
                    
                    binary_name = result.split("/")[-1]
                    
                    # Search order for binary:
                    # 1. Directly in bin/
                    # 2. In Resources/bin/ (Electron specific)
                    # 3. Path relative to current executable
                    possible_paths = [
                        PROJECT_ROOT / "bin" / binary_name,
                        PROJECT_ROOT / "Resources" / "bin" / binary_name,
                        Path(sys.executable).parent / binary_name
                    ]
                    
                    for p in possible_paths:
                        if p.exists():
                            logger.info(f"[MCP] Redirected {binary_name} -> {p}")
                            return str(p)

            return result

        name_overrides = {
            "whisper-stt": "whisper",
        }

        servers = raw_config.get("mcpServers", {})
        for server_name, server_config in servers.items():
            # Skip comments and disabled servers
            if server_name.startswith("_") or server_config.get("disabled", False):
                continue

            yaml_key = name_overrides.get(server_name, server_name.replace("-", "_"))
            if config.get(f"mcp.{yaml_key}.enabled", True) is False:
                continue

            missing_env: List[str] = []

            # Substitute environment variables in env section
            if "env" in server_config:
                env_vars: Dict[str, Any] = {}
                for key, value in (server_config.get("env") or {}).items():
                    env_vars[key] = _substitute_placeholders(value, missing_env)
                server_config["env"] = env_vars

            # Substitute environment variables in args section
            if "args" in server_config and isinstance(server_config.get("args"), list):
                new_args: List[Any] = []
                for arg in server_config.get("args") or []:
                    new_args.append(_substitute_placeholders(arg, missing_env))
                server_config["args"] = new_args

            # Substitute environment variables in command
            if "command" in server_config:
                server_config["command"] = _substitute_placeholders(
                    server_config["command"], missing_env
                )

            if missing_env:
                server_config["_missing_env"] = sorted(set(missing_env))

            processed["mcpServers"][server_name] = server_config

        return processed

    def _kill_orphans(self, server_name: str, command: str):
        """Kills any lingering processes that match the server command-args signature."""
        try:
            import subprocess
            # We look for processes running the same command/module
            search_str = ""
            if "src.mcp_server" in command:
                search_str = command.split(".")[-1]
            elif "mcp-server-" in command:
                search_str = command.split("/")[-1]
            else:
                search_str = server_name

            if not search_str: return

            logger.info(f"[MCP] Cleaning up orphan processes for {server_name} (search: {search_str})")
            subprocess.run(["pkill", "-9", "-f", search_str], capture_output=True)
            # Give OS a moment to release ports/files
            import time
            time.sleep(0.5)
        except Exception as e:
            logger.debug(f"Orphan cleanup failed for {server_name}: {e}")


    async def get_session(self, server_name: str) -> Optional[ClientSession]:
        """Get or create a persistent session for the server"""
        if ClientSession is None or StdioServerParameters is None or stdio_client is None:
            logger.error("MCP Python package is not installed; MCP features are unavailable")
            return None

        async with self._lock:
            if server_name in self.sessions:
                return self.sessions[server_name]

            server_config = self.config.get("mcpServers", {}).get(server_name)
            if not server_config:
                logger.error(f"Server {server_name} not configured")
                return None

            try:
                return await self._connect_server(server_name, server_config)
            except Exception as e:
                logger.warning(f"get_session: failed to connect to {server_name}: {e}")
                return None

    async def _connect_server(
        self, server_name: str, config: Dict[str, Any]
    ) -> Optional[ClientSession]:
        """Establish a new connection to an MCP server"""
        default_timeout = float(
            self.config.get("mcpServers", {}).get("_defaults", {}).get("connect_timeout", 30.0)
        )
        connect_timeout = float(config.get("connect_timeout", default_timeout))

        missing_env = config.get("_missing_env")
        if missing_env:
            logger.error(
                f"Missing required environment variables for MCP server '{server_name}': {missing_env}. "
                "Set them in ~/.config/atlastrinity/.env or your shell environment."
            )
            return None
        command = config.get("command")
        args = config.get("args", [])
        env = os.environ.copy()
        env.update(config.get("env", {}))
        
        # Ensure PYTHONPATH includes project root so that 'src.mcp_server' can be resolved
        from .config import PROJECT_ROOT
        root_path = str(PROJECT_ROOT)
        current_pp = env.get("PYTHONPATH", "")
        if root_path not in current_pp:
            env["PYTHONPATH"] = f"{root_path}{os.pathsep}{current_pp}" if current_pp else root_path
        
        logger.debug(f"[MCP] PYTHONPATH for {server_name}: {env.get('PYTHONPATH')}")

        # Resolve command path
        if command == "python3" or command == "python":
            command = sys.executable
        elif command == "npx":
            npx_path = shutil.which("npx")
            if npx_path:
                command = npx_path
            else:
                # Standard fallbacks for macOS
                fallbacks = [
                    "/opt/homebrew/bin/npx",  # Apple Silicon
                    "/usr/local/bin/npx",  # Intel
                    "/opt/homebrew/opt/node@22/bin/npx",
                    os.path.expanduser("~/.nvm/current/bin/npx"),
                ]
                for fb in fallbacks:
                    if os.path.exists(fb):
                        command = fb
                        break

            # === PRE-FLIGHT: verify package versions for npx/bunx invocations ===
            # If the first arg looks like 'package@version', ensure the version exists
            # in the registry before spawning the external command.
            if len(args) > 0 and not check_package_arg_for_tool(args[0], tool_cmd=command):
                logger.error(
                    f"Requested package '{args[0]}' for command '{command}' does not exist or version not available in registry. Aborting start for this MCP."
                )
                return None
        elif command == "bunx":
            bunx_path = shutil.which("bunx")
            if bunx_path:
                command = bunx_path
            else:
                # Standard fallbacks
                fallbacks = [
                    os.path.expanduser("~/.bun/bin/bunx"),
                    "/usr/local/bin/bunx",
                    "/opt/homebrew/bin/bunx",
                ]
                for fb in fallbacks:
                    if os.path.exists(fb):
                        command = fb
                        break

        server_params = StdioServerParameters(command=command, args=args, env=env)

        # If a connection task already exists, wait for its session future
        if server_name in self._connection_tasks:
            fut = self._session_futures.get(server_name)
            if fut is None:
                return None
            try:
                session = await asyncio.wait_for(fut, timeout=connect_timeout)
                return session
            except Exception as e:
                logger.error(f"Existing connection for {server_name} failed to initialize: {e}")
                return None

        # Create per-server close event and a future that will be completed
        close_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        session_future: asyncio.Future = loop.create_future()

        async def connection_runner():
            try:
                # Clean up previous instances before starting
                # (Added to address user feedback about too many servers)
                # self._kill_orphans(server_name, command)
                # Wait, better not to kill during connect_server if it's already managed.
                # Only do it in restart_server.
                
                logger.info(f"Connecting to MCP server: {server_name}...")
                logger.debug(f"[MCP] Command: {command}, Args: {args}")
                async with stdio_client(server_params) as (read, write):
                    # Define logging callback for this server
                    async def handle_log(params: Any):
                        # Extract level and data
                        level_str = getattr(params, "level", "info").lower()
                        data = getattr(params, "data", "")
                        
                        # Format message
                        msg = data
                        
                        # Log to main brain logger - DISABLED to avoid duplication with Orchestrator callback
                        # if level_str in ["debug", "trace"]:
                        #     logger.debug(msg)
                        # elif level_str in ["warning", "warn"]:
                        #     logger.warning(msg)
                        # elif level_str in ["error", "critical", "fatal"]:
                        #     logger.error(msg)
                        # else:
                        #     logger.info(msg)

                        # Notify callbacks (e.g. for WebSocket limits)
                        for cb in self._log_callbacks:
                            try:
                                if asyncio.iscoroutinefunction(cb):
                                    # We use a wrapper for the task to catch internal callback errors
                                    async def safe_cb_wrapper():
                                        try:
                                            await cb(msg, server_name, level_str)
                                        except Exception as e:
                                            logger.error(f"[MCP] Log callback internal error ({server_name}): {e}")
                                    
                                    asyncio.create_task(safe_cb_wrapper())
                                else:
                                    cb(msg, server_name, level_str)
                            except Exception as e:
                                logger.error(f"[MCP] Log callback dispatch error ({server_name}): {e}")

                    async with ClientSession(read, write, logging_callback=handle_log) as session:
                        await session.initialize()
                        
                        # store session usable by other tasks
                        self.sessions[server_name] = session
                        if not session_future.done():
                            session_future.set_result(session)
                        logger.info(f"Connected to {server_name}")
                        # keep the connection alive until asked to close
                        await close_event.wait()
            except Exception as e:
                if not session_future.done():
                    session_future.set_exception(e)
                logger.error(
                    f"Failed to run connection for {server_name}: {type(e).__name__}: {e}",
                    exc_info=True,
                )
            finally:
                # ensure cleanup from the connection's own task
                self.sessions.pop(server_name, None)
                self._connection_tasks.pop(server_name, None)
                self._close_events.pop(server_name, None)
                self._session_futures.pop(server_name, None)
                logger.info(f"Connection task for {server_name} exited")

        task = asyncio.create_task(connection_runner(), name=f"mcp-{server_name}")
        self._connection_tasks[server_name] = task
        self._close_events[server_name] = close_event
        self._session_futures[server_name] = session_future

        try:
            session = await asyncio.wait_for(session_future, timeout=connect_timeout)
            return session
        except Exception as e:
            # If we couldn't initialize, ask runner to exit and await it
            logger.error(f"Failed to connect to {server_name}: {type(e).__name__}: {e}")
            logger.debug(f"[MCP] Command: {command}, Args: {args}, Env keys: {list(env.keys())}")
            try:
                close_event.set()
                await task
            except Exception:
                pass
            # Re-raise to allow callers (eg. restart_server) to decide on retry behavior
            raise

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any] = None
    ) -> Any:
        """Call a tool on a specific server"""
        session = await self.get_session(server_name)
        if not session:
            return {"error": f"Could not connect to server {server_name}"}

        try:
            result = await session.call_tool(tool_name, arguments or {})

            # Safety: Truncate large outputs to prevent OOM/Context overflow
            if hasattr(result, "content") and isinstance(result.content, list):
                for item in result.content:
                    if hasattr(item, "text") and isinstance(item.text, str):
                        if len(item.text) > 100 * 1024 * 1024:  # 100MB limit
                            item.text = item.text[: 100 * 1024 * 1024] + "\n... [TRUNCATED DUE TO SIZE] ..."
                            logger.warning(f"Truncated large output from {server_name}.{tool_name}")

            return result
        except Exception as e:
            # Enhanced error formatting for ExceptionGroup
            error_msg = str(e)
            if hasattr(e, "exceptions") and e.exceptions: # ExceptionGroup
                error_msg = f"{type(e).__name__}: {', '.join([str(x) for x in e.exceptions])}"

            logger.error(f"Error calling tool {server_name}.{tool_name}: {error_msg}")

            # Special handling for vibe server - try to auto-enable on errors
            if server_name == "vibe":
                try:
                    # Check if actually disabled
                    vibe_cfg = self.config.get("mcpServers", {}).get("vibe", {})
                    if vibe_cfg.get("disabled"):
                        logger.warning("[MCP] Vibe tool failed and server is disabled in config")
                except Exception as config_err:
                    logger.error(f"[MCP] Failed to check vibe config: {config_err}")

            # If connection died, try to reconnect once
            if any(kw in error_msg for kw in ["Connection closed", "Broken pipe", "ClosedResourceError", "unhandled errors in a TaskGroup"]):
                logger.warning(f"Connection lost to {server_name}, attempting reconnection...")
                async with self._lock:
                    if server_name in self.sessions:
                        del self.sessions[server_name]
                    if server_name in self._connection_tasks:
                        # Cancel existing task to allow fresh start
                        task = self._connection_tasks.pop(server_name)
                        task.cancel()

                session = await self.get_session(server_name)
                if session:
                    try:
                        return await session.call_tool(tool_name, arguments or {})
                    except Exception as retry_e:
                        return {"error": f"Retry failed: {str(retry_e)}"}

            return {"error": error_msg}

    async def dispatch_tool(
        self, 
        tool_name: Optional[str], 
        arguments: Dict[str, Any] = None,
        explicit_server: Optional[str] = None,
        allow_fallback: bool = True
    ) -> Any:
        """
        Unified entry point for tool calls with resolution, normalization, and intelligent fallback.
        
        Args:
            tool_name: Tool name (can include server as dot notation)
            arguments: Tool arguments
            explicit_server: Explicitly specified server
            allow_fallback: If True, attempt macOS-use fallback on failure
        """
        result = await self.dispatcher.resolve_and_dispatch(tool_name, arguments or {}, explicit_server)
        
        # Intelligent fallback: If failed and allow_fallback, try macOS-use equivalent
        if allow_fallback and isinstance(result, dict) and result.get("error"):
            # Check if this wasn't already a macOS-use call
            if explicit_server != "macos-use" and not (tool_name or "").startswith("macos-use"):
                # Try to find macOS-use equivalent
                fallback_tool = self._get_macos_equivalent(tool_name or "")
                if fallback_tool:
                    logger.warning(
                        f"[MCP] Primary tool failed: {tool_name}. "
                        f"Attempting fallback to macos-use.{fallback_tool}"
                    )
                    try:
                        result = await self.call_tool("macos-use", fallback_tool, arguments or {})
                        logger.info(f"[MCP] Fallback successful: macos-use.{fallback_tool}")
                    except Exception as e:
                        logger.error(f"[MCP] Fallback also failed: {e}")
        
        return result
    
    def _get_macos_equivalent(self, tool_name: str) -> Optional[str]:
        """Find macOS-use equivalent for a given tool."""
        tool_lower = tool_name.lower()
        
        # Mapping of common tools to macOS-use equivalents
        equivalents = {
            "fetch": "macos-use_fetch_url",
            "fetch_url": "macos-use_fetch_url",
            "get_time": "macos-use_get_time",
            "time": "macos-use_get_time",
            "screenshot": "macos-use_take_screenshot",
            "terminal": "execute_command",
            "execute": "execute_command",
            "run_command": "execute_command",
            "search": "macos-use_spotlight_search",
            "spotlight": "macos-use_spotlight_search",
        }
        
        return equivalents.get(tool_lower)

    async def list_tools(self, server_name: str) -> List[Any]:
        """List available tools for a server"""
        session = await self.get_session(server_name)
        if not session:
            logger.warning(f"[MCP] Could not get session for {server_name}")
            return []

        try:
            result = await session.list_tools()
            return result.tools
        except Exception as e:
            # FIX: Handle ClosedResourceError by reconnecting
            if "ClosedResourceError" in str(e) or "Connection closed" in str(e):
                logger.warning(f"[MCP] Connection lost during list_tools for {server_name}, reconnecting...")
                async with self._lock:
                    if server_name in self.sessions:
                        del self.sessions[server_name]
                
                # Try to get fresh session
                session = await self.get_session(server_name)
                if session:
                    try:
                        result = await session.list_tools()
                        return result.tools
                    except Exception as retry_e:
                        logger.error(f"[MCP] Retry list_tools failed for {server_name}: {retry_e}")
            
            logger.error(
                f"Error listing tools for {server_name}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return []

    async def health_check(self, server_name: str) -> bool:
        """
        Check if a server is healthy.
        Returns True if server responds, False otherwise.
        """
        session = self.sessions.get(server_name)
        if not session:
            return False

        try:
            # Try to list tools as a health check
            await session.list_tools()
            return True
        except Exception as e:
            # Special handling for vibe server - try to auto-enable on errors
            if server_name == "vibe":
                logger.warning(f"[MCP] Vibe server unhealthy, attempting auto-enable: {e}")
                # Try to enable vibe via self-healing
                try:
                    from .config_loader import config  # noqa: E402

                    if not config.get("mcp.vibe.enabled", False):
                        logger.info("[MCP] Auto-enabling vibe server due to health check failure")
                        # Update config to enable vibe
                        # This would require config reload - for now just log
                        logger.info("[MCP] Consider enabling vibe in config.yaml")
                except Exception as config_err:
                    logger.error(f"[MCP] Failed to check vibe config: {config_err}")
            return False

    async def restart_server(self, server_name: str) -> bool:
        """
        Force restart a server connection with retry/backoff and concurrency limits.
        """
        logger.warning(f"[MCP] Restarting server: {server_name}")

        async with self._restart_semaphore:
            server_config = self.config.get("mcpServers", {}).get(server_name)
            async with self._lock:
                # Signal old connection to close
                if server_name in self._close_events:
                    try:
                        self._close_events[server_name].set()
                    except Exception:
                        pass
                    # await the connection task to exit
                    task = self._connection_tasks.get(server_name)
                    if task:
                        try:
                            await task
                        except Exception:
                            pass
                # Remove any remaining session reference
                if server_name in self.sessions:
                    del self.sessions[server_name]
                
                # Proactive cleanup of lingering child processes
                cmd = server_config.get("command", "") if server_config else ""
                self._kill_orphans(server_name, cmd)

            # Treat an empty dict as a valid server configuration. Only fail if the entry is missing entirely.
            if server_config is None:
                logger.error(f"No configuration for server {server_name}")
                return False

            last_exc = None
            for attempt in range(1, self._max_restart_attempts + 1):
                try:
                    session = await self._connect_server(server_name, server_config)
                    if session:
                        logger.info(
                            f"[MCP] Server {server_name} restarted successfully (attempt {attempt})"
                        )
                        return True
                except Exception as e:
                    last_exc = e
                    # If this is a resource fork error (EAGAIN), do a backoff and retry
                    try:
                        import errno as _errno  # noqa: E402

                        if isinstance(e, OSError) and getattr(e, "errno", None) == _errno.EAGAIN:
                            wait = self._restart_backoff_base * (2 ** (attempt - 1))
                            logger.warning(
                                f"[MCP] Spawn EAGAIN for {server_name}, backing off {wait:.1f}s (attempt {attempt})"
                            )
                            await asyncio.sleep(wait)
                            continue
                    except Exception:
                        pass

                    # For other exceptions, small backoff then retry
                    wait = min(10.0, self._restart_backoff_base * (2 ** (attempt - 1)))
                    logger.warning(
                        f"[MCP] Restart attempt {attempt} for {server_name} failed: {type(e).__name__}: {e}. Retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue

            logger.error(
                f"[MCP] Failed to restart {server_name} after {self._max_restart_attempts} attempts: {last_exc}"
            )
            return False

    async def health_check_loop(self, interval: int = 60):
        """
        Background task that monitors server health.
        Automatically restarts failed servers.

        Args:
            interval: Seconds between health checks
        """
        logger.info(f"[MCP] Starting health check loop (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)

                # Check all connected servers
                for server_name in list(self.sessions.keys()):
                    is_healthy = await self.health_check(server_name)

                    if not is_healthy:
                        logger.warning(f"[MCP] Server {server_name} unhealthy, restarting...")
                        success = await self.restart_server(server_name)
                        if success:
                            logger.info(f"[MCP] Server {server_name} restarted successfully")
                        else:
                            logger.error(f"[MCP] Failed to restart {server_name}")

            except asyncio.CancelledError:
                logger.info("[MCP] Health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"[MCP] Health check error: {e}")

    def start_health_monitoring(self, interval: int = 60):
        """Start the health check background task."""
        self._health_task = asyncio.create_task(self.health_check_loop(interval))
        return self._health_task

    def get_status(self) -> Dict[str, Any]:
        """Get status of all servers including coverage statistics."""
        status = {
            "connected_servers": list(self.sessions.keys()),
            "configured_servers": list(self.config.get("mcpServers", {}).keys()),
            "session_count": len(self.sessions),
        }
        
        # Add coverage statistics
        try:
            coverage = self.dispatcher.get_coverage_stats()
            status["coverage"] = coverage
        except Exception as e:
            logger.warning(f"[MCP] Could not get coverage stats: {e}")
        
        return status

    def register_log_callback(self, callback):
        """Register a callback to receive log notifications from servers."""
        if callback not in self._log_callbacks:
            self._log_callbacks.append(callback)

    def unregister_log_callback(self, callback):
        """Unregister a log callback."""
        if callback in self._log_callbacks:
            self._log_callbacks.remove(callback)

    # ═══════════════════════════════════════════════════════════════════════════
    #                     LAZY INITIALIZATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_server_catalog_without_connection(self) -> str:
        """
        Returns full server catalog from registry WITHOUT connecting to any servers.
        LLM uses this to decide which servers to initialize.
        
        This is FAST: no network calls, pure static data from mcp_registry.
        """
        from .mcp_registry import get_server_catalog_for_prompt
        return get_server_catalog_for_prompt(include_key_tools=True)

    async def ensure_servers_connected(self, server_names: List[str]) -> Dict[str, bool]:
        """
        Lazily initialize only the specified servers.
        
        Args:
            server_names: List of server names to connect
            
        Returns:
            Dict mapping server_name -> success (True/False)
        """
        results = {}
        for name in server_names:
            if name in self.sessions:
                # Already connected
                results[name] = True
                logger.debug(f"[MCP] Server {name} already connected")
            else:
                try:
                    session = await self.get_session(name)
                    results[name] = session is not None
                    if session:
                        logger.info(f"[MCP] Lazily initialized: {name}")
                    else:
                        logger.warning(f"[MCP] Failed to initialize: {name}")
                except Exception as e:
                    logger.error(f"[MCP] Error initializing {name}: {e}")
                    results[name] = False
        return results

    def get_available_servers(self) -> List[str]:
        """Get list of all configured (available) server names."""
        return list(self.config.get("mcpServers", {}).keys())

    async def shutdown(self):
        """
        Coordinated shutdown of all MCP server connections.
        Ensures all tasks are cancelled and orphan processes are killed.
        """
        logger.info("[MCP] Initiating shutdown of all server connections...")
        
        # 1. Signal all connections to close gracefully
        async with self._lock:
            server_names = list(self._close_events.keys())
            for name in server_names:
                try:
                    self._close_events[name].set()
                except Exception:
                    pass
        
        # 2. Give tasks a moment to exit
        if self._connection_tasks:
            await asyncio.sleep(0.5)
            
        # 3. Kill lingering tasks and processes
        async with self._lock:
             for name, task in list(self._connection_tasks.items()):
                 if not task.done():
                     logger.warning(f"[MCP] Force cancelling connection task for {name}")
                     task.cancel()
             
             # Final pkill for any common MCP signatures
             try:
                 import subprocess
                 # Targeted kills for known servers
                 sigs = ["mcp-server", "macos-use", "vibe_server", "vibe", "npx", "bunx"]
                 for sig in sigs:
                     subprocess.run(["pkill", "-9", "-f", sig], capture_output=True)
             except Exception:
                 pass
             
             self.sessions.clear()
             self._connection_tasks.clear()
             self._close_events.clear()
             self._session_futures.clear()

        logger.info("[MCP] Shutdown complete.")


mcp_manager = MCPManager()

    def get_connected_servers(self) -> List[str]:
        """Get list of currently connected server names."""
        return list(self.sessions.keys())

    async def get_mcp_catalog(self, connected_only: bool = False) -> str:
        """
        Generates a concise catalog of all configured MCP servers and their roles.
        
        Args:
            connected_only: If True, only shows already connected servers (fast).
                           If False, tries to connect to all servers (slower).
        
        Returns:
            Formatted catalog string for LLM consumption.
        """
        # OPTIMIZATION: Use static registry for fast catalog generation
        if connected_only:
            from .mcp_registry import SERVER_CATALOG
            
            catalog = "MCP SERVER CATALOG (Connected Servers):\n"
            connected = self.get_connected_servers()
            
            for name in connected:
                info = SERVER_CATALOG.get(name, {})
                desc = info.get("description", "Native capability")
                catalog += f"[CONNECTED] {name}: {desc}\n"
            
            if not connected:
                catalog += "(No servers currently connected)\n"
            
            catalog += "\nUse get_server_catalog_without_connection() to see all available servers."
            return catalog
        
        # Original behavior: try to fetch tools from all servers
        catalog = "MCP SERVER CATALOG (Available Realms):\n"
        configured_servers = self.config.get("mcpServers", {})

        # Prepare tasks for fetching tools in parallel
        tasks = []
        server_names = []

        for server_name, server_cfg in configured_servers.items():
            if server_name.startswith("_"):
                continue

            # Only attempt to fetch tools if server is not disabled
            if not server_cfg.get("disabled", False):
                tasks.append(self.list_tools(server_name))
                server_names.append(server_name)

        # Fetch all tools with a timeout to prevent hanging
        try:
            # 2-second timeout for catalog generation is sufficient; if it's slower, we skip tool details
            all_tools_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=2.0
            )
        except Exception:
            all_tools_results = [[] for _ in tasks]  # Fallback to empty lists on overall timeout

        # Map results back to servers
        server_tools_map = {}
        for i, name in enumerate(server_names):
            res = all_tools_results[i]
            if isinstance(res, list):
                server_tools_map[name] = [getattr(t, "name", str(t)) for t in res]
            else:
                server_tools_map[name] = []

        for server_name, server_cfg in configured_servers.items():
            if server_name.startswith("_"):
                continue

            description = server_cfg.get("description") or "Native or custom capability."
            status = "CONNECTED" if server_name in self.sessions else "AVAILABLE"

            tool_names = server_tools_map.get(server_name, [])
            tool_str = ""
            if tool_names:
                # Limit to first 10 tools to keep context small, or all if short names
                joined_tools = ", ".join(tool_names[:10])
                if len(tool_names) > 10:
                    joined_tools += ", ..."
                tool_str = f" (Tools: {joined_tools})"

            catalog += f"[{status}] {server_name}: {description}{tool_str}\n"

        catalog += "\nTo see specific tool schemas, use 'inspect_mcp_server' (or it will be done automatically by Tetyana)."
        return catalog

    async def get_tools_summary(self) -> str:
        """
        Generates a detailed summary of all available tools across all servers,
        including their input schemas (arguments) for precise LLM mapping.
        """
        import json  # noqa: E402

        summary = "AVAILABLE MCP TOOLS (Full Specs):\n"
        configured_servers = [s for s in self.config.get("mcpServers", {}) if not s.startswith("_")]

        async def fetch_tools(server_name):
            try:
                tools = await asyncio.wait_for(self.list_tools(server_name), timeout=5.0)
                return server_name, tools
            except Exception:
                return server_name, []

        results = await asyncio.gather(*[fetch_tools(s) for s in configured_servers])

        for server_name, tools in results:
            if tools:
                summary += f"\n--- SERVER: {server_name} ---\n"
                for tool in tools:
                    name = getattr(tool, "name", str(tool))
                    desc = getattr(tool, "description", "No description")
                    schema = getattr(tool, "inputSchema", {})

                    # Format as compact JSON for the LLM
                    schema_str = json.dumps(schema, ensure_ascii=False) if schema else "{}"

                    summary += f"- {name}: {desc}\n"
                    if schema:
                        # Extract parameter names and types for quick reference
                        params = schema.get("properties", {})
                        if params:
                            param_list = [
                                f"{p} ({v.get('type', 'any')})" for p, v in params.items()
                            ]
                            summary += f"  Args: {', '.join(param_list)}\n"
                        summary += f"  Schema: {schema_str}\n"
            else:
                summary += f"- {server_name} (No tools responsive)\n"

        return summary



    async def query_db(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a raw SQL query (for debugging/self-healing)"""
        if not db_manager.available:
            try:
                await db_manager.initialize()
            except Exception as e:
                return [{"error": f"Database initialization failed: {e}"}]
        
        if not db_manager.available:
            return [{"error": "Database not available"}]
        
        try:
            async with await db_manager.get_session() as session:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
        except Exception as e:
            return [{"error": str(e)}]

    async def cleanup(self):
        """Close all connections"""
        logger.info("Closing all MCP connections...")

        # Cancel health check if running
        if hasattr(self, "_health_task"):
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Signal all connection tasks to close and wait for them
        tasks = list(self._connection_tasks.values())
        for name, ev in list(self._close_events.items()):
            try:
                ev.set()
            except Exception:
                pass
        for task in tasks:
            try:
                await task
            except Exception:
                pass
        # Clear sessions and internal maps
        self.sessions.clear()
        self._connection_tasks.clear()
        self._close_events.clear()
        self._session_futures.clear()


# Global instance
mcp_manager = MCPManager()
