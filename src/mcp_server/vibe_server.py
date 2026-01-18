"""
Vibe MCP Server - CLI Mode Integration

This server wraps the Vibe CLI (Mistral-powered) in programmatic mode,
enabling full logging visibility in the Electron app.

Key Features:
- Uses `vibe -p "prompt" --output json` for programmatic execution
- All output is structured JSON for easy parsing
- Full logging of Vibe actions visible in UI logs
- Self-healing and debugging capabilities

Author: AtlasTrinity Team
Updated: 2026-01-14
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import uuid
import re
import pty
from typing import Any, Dict, List, Optional, Tuple, Pattern
from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP, Context

# ANSI escape code regex
ANSI_ESCAPE: Pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE.sub('', text)


# Setup logging for visibility in Electron app
logger = logging.getLogger("vibe_mcp")
logger.setLevel(logging.INFO)

# Standard path for AtlasTrinity logs
try:
    _log_dir = Path.home() / ".config" / "atlastrinity" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    
    # Emergency debug log
    fh = logging.FileHandler(_log_dir / "vibe_server_debug.log", mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    
    import sys
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[VIBE_MCP] %(message)s"))
    logger.addHandler(sh)
except Exception:
    pass

try:
    from .config_loader import get_config_value, CONFIG_ROOT, PROJECT_ROOT

    VIBE_BINARY = get_config_value("mcp.vibe", "binary", "vibe")
    DEFAULT_TIMEOUT_S = float(get_config_value("mcp.vibe", "timeout_s", 3600))
    # Increased for large log analysis
    MAX_OUTPUT_CHARS = int(get_config_value("mcp.vibe", "max_output_chars", 500000))
    DISALLOW_INTERACTIVE = bool(get_config_value("mcp.vibe", "disallow_interactive", True))
    
    # Resolve global vibe_workspace (handled by get_config_value resolution)
    VIBE_WORKSPACE = get_config_value("mcp.vibe", "workspace", str(CONFIG_ROOT / "vibe_workspace"))
except Exception:
    VIBE_BINARY = "vibe"
    DEFAULT_TIMEOUT_S = 3600.0
    MAX_OUTPUT_CHARS = 500000 
    DISALLOW_INTERACTIVE = True
    VIBE_WORKSPACE = str(Path.home() / ".config" / "atlastrinity" / "vibe_workspace")
    CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
    PROJECT_ROOT = Path(__file__).parent.parent.parent


PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

# System root for self-healing (where the AtlasTrinity source code lives)
def _get_system_root():
    try:
        from .config_loader import load_mcp_config, _substitute_placeholders
        config_path = CONFIG_ROOT / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
                raw_path = full_cfg.get("system", {}).get("repository_path", str(PROJECT_ROOT))
                return _substitute_placeholders(raw_path)
    except Exception:
        pass
    return str(PROJECT_ROOT)

SYSTEM_ROOT = _get_system_root()
LOG_DIR = str(CONFIG_ROOT / "logs")

# Vibe session directory (default CLI location)
VIBE_SESSION_DIR = Path.home() / ".vibe" / "logs" / "session"

# Global instructions directory for large prompts
INSTRUCTIONS_DIR = str(Path(VIBE_WORKSPACE) / "instructions")


def _cleanup_old_instructions(max_age_hours: int = 24):
    """Remove instruction files older than max_age_hours."""
    instructions_path = Path(INSTRUCTIONS_DIR)
    if not instructions_path.exists():
        return 0
    
    now = datetime.now()
    cleaned = 0
    for f in instructions_path.glob("vibe_instructions_*.md"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if (now - mtime).total_seconds() > max_age_hours * 3600:
                f.unlink()
                logger.info(f"[VIBE] Cleaned up old instruction file: {f.name}")
                cleaned += 1
        except Exception as e:
            logger.debug(f"[VIBE] Cleanup failed for {f.name}: {e}")
    return cleaned


# CLI-only subcommands (no TUI)
ALLOWED_SUBCOMMANDS = {
    "list-editors",
    "list-modules",
    "run",
    "enable",
    "disable",
    "install",
    "smart-plan",
    "ask",
    "agent-reset",
    "agent-on",
    "agent-off",
    "vibe-status",
    "vibe-continue",
    "vibe-cancel",
    "vibe-help",
    "eternal-engine",
    "screenshots",
}

# DB Configuration (Shared with AtlasTrinity brain)
DATABASE_URL = get_config_value("database", "url", os.getenv("DATABASE_URL", "postgresql+asyncpg://dev:postgres@localhost/atlastrinity_db"))

# Subcommands that are BLOCKED (interactive TUI)
BLOCKED_SUBCOMMANDS = {
    "tui",
    "agent-chat",  # Use vibe_prompt instead for programmatic mode
    "self-healing-status",  # TUI mode, use vibe_prompt for queries
    "self-healing-scan",  # TUI mode, use vibe_prompt for queries
}


server = FastMCP("vibe")

# Cleanup old instruction files on startup
try:
    _cleaned = _cleanup_old_instructions(max_age_hours=24)
    if _cleaned > 0:
        logger.info(f"[VIBE] Cleaned up {_cleaned} old instruction files on startup")
except Exception:
    pass


def _truncate(text: str) -> str:
    """Truncate text to max output chars with indicator."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [TRUNCATED - Output exceeded 500KB] ..."


def _resolve_vibe_binary() -> str:
    """Resolve the path to the Vibe CLI binary."""
    # Try expanded user path first (common for .local/bin)
    expanded = os.path.expanduser("~/.local/bin/vibe")
    if os.path.exists(expanded):
        return expanded
        
    # Check config
    if os.path.isabs(VIBE_BINARY) and os.path.exists(VIBE_BINARY):
        return VIBE_BINARY
        
    # Check PATH
    path_res = shutil.which(VIBE_BINARY)
    if path_res:
        return path_res
        
    return VIBE_BINARY


def _parse_stack_trace(error_msg: str) -> Optional[Dict[str, str]]:
    """Extract file path and line number from python stack trace."""
    # Simple regex for: File "path/to/file.py", line 123
    match = re.search(r'File "([^"]+)", line (\d+)', error_msg)
    if match:
        return {"file": match.group(1), "line": match.group(2)}
    return None


def _prepare_prompt_arg(prompt: str, cwd: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Prepare the prompt argument. If too long, offload to a temporary file.
    Returns (final_prompt_arg, file_path_to_clean).
    
    Note: Instruction files are ALWAYS stored in INSTRUCTIONS_DIR (global folder),
    regardless of the cwd parameter. The cwd only affects where Vibe CLI executes.
    """
    if len(prompt) <= 2000:
        return prompt, None

    try:
        # Always use global instructions directory, NOT cwd
        os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)
            
        prompt_file = f"vibe_instructions_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}.md"
        prompt_path = os.path.join(INSTRUCTIONS_DIR, prompt_file)
        
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("# INSTRUCTIONS FOR VIBE AGENT\n\n")
            f.write(prompt)
        
        logger.info(f"[VIBE] Large prompt ({len(prompt)} chars) offloaded to {prompt_path}")
        # Return full path so Vibe can find it from any working directory
        return f"Please read and execute the instructions detailed in the file: {prompt_path}", prompt_path

    except Exception as e:
        logger.warning(f"[VIBE] Failed to write prompt file: {e}")
        # Fallback
        if len(prompt) > 10000:
            return prompt[:10000] + "\n...[TRUNCATED]", None
        return prompt, None



async def _run_vibe(
    argv: List[str],
    cwd: Optional[str],
    timeout_s: float,
    extra_env: Optional[Dict[str, str]],
    ctx: Any = None,
) -> Dict[str, Any]:
    """Execute Vibe CLI command and return structured result with real-time logging."""
    env = os.environ.copy()
    
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    async def safe_notify(msg: str, is_error: bool = False):
        if not ctx: 
            print(f"[VIBE_DEBUG] No ctx for: {msg}", file=sys.stderr)
            return
        try:
            print(f"[VIBE_DEBUG] Notifying: {msg}", file=sys.stderr)
            if is_error:
                await ctx.error(msg)
            else:
                await ctx.info(msg)
        except Exception as e:
            print(f"[VIBE_DEBUG] safe_notify failed: {e}", file=sys.stderr)
            pass
    
    # Wrapper implementation: Use vibe_runner.py to handle PTY isolation
    runner_script = os.path.join(os.path.dirname(__file__), "vibe_runner.py")
    
    # Prepend python and runner script to argv
    # argv was [vibe_path, args...] (e.g. /home/user/.local/bin/vibe -p ...)
    # wrapper takes [vibe_binary, args...] as arguments.
    wrapper_argv = [sys.executable, runner_script] + argv
    
    msg_start = f"⚡ [VIBE-LIVE] Initializing... (Timeout: {timeout_s}s)"
    logger.info(msg_start)
    asyncio.create_task(safe_notify(msg_start))

    try:
        print(f"[VIBE_DEBUG] Executing (Wrapper): {wrapper_argv}", file=sys.stderr)
        
        # Run the wrapper as a standard subprocess (wrapper handles PTY)
        process = await asyncio.create_subprocess_exec(
            *wrapper_argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, 
            stdin=asyncio.subprocess.DEVNULL 
        )

        stdout_chunks = []
        stderr_chunks = []

        # Shared state for heartbeat
        last_activity = [asyncio.get_event_loop().time()]

        async def read_stream(stream, chunks, prefix):
            buffer = ""
            while True:
                # Read chunks
                data = await stream.read(512)
                if not data:
                    break
                
                # Update activity timestamp
                last_activity[0] = asyncio.get_event_loop().time()

                try:
                    text_chunk = data.decode(errors='replace')
                except:
                    text_chunk = str(data)
                
                chunks.append(text_chunk)
                buffer += text_chunk
                
                lines = []
                if "\n" in buffer:
                    lines = buffer.split("\n")
                    buffer = lines.pop()
                    
                for raw_text in lines:
                    if not raw_text.strip():
                        continue
                    
                    # LOGGING LOGIC
                    # Wrapper ensures mostly clean JSON, but we verify
                    try:
                        data_json = json.loads(raw_text)
                        
                        role = data_json.get("role", "")
                        thoughts = data_json.get("reasoning_content") or data_json.get("thought") or ""
                        content = data_json.get("content") or ""
                        tool_calls = data_json.get("tool_calls")
                        
                        # Log role-based message
                        if role:
                            msg = f"📨 [VIBE-MSG] Role: {role}"
                            logger.info(msg)
                        
                        if thoughts:
                            snippet = thoughts[:800] + ("..." if len(thoughts) > 800 else "")
                            msg = f"🧠 [VIBE-THOUGHT] {snippet}"
                            logger.info(msg)
                            asyncio.create_task(safe_notify(msg))
                        
                        if tool_calls:
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                f_name = func.get("name", "unknown_tool")
                                msg = f"🛠️ [VIBE-ACTION] Using tool: {f_name}"
                                logger.info(msg)
                                asyncio.create_task(safe_notify(msg))
                        
                        if content:
                            snippet = content.strip().replace("\n", " ")[:500]
                            if snippet:
                                msg = f"📝 [VIBE-GEN] {snippet}"
                                logger.info(msg)
                                asyncio.create_task(safe_notify(msg))
                                
                    except json.JSONDecodeError:
                        # Non-JSON content (wrapper passed it through?)
                        if len(raw_text) < 1000:
                             msg = f"⚡ [VIBE-STATUS] {raw_text}"
                             logger.info(msg)
                             asyncio.create_task(safe_notify(msg))

        # Heartbeat task with informative progress messages
        async def heartbeat_worker():
            ticks = 0
            phases = [
                "🔍 Analyzing request...",
                "🧠 Deep thinking in progress...",
                "💭 Reasoning through the problem...",
                "📝 Formulating response...",
                "🔧 Planning tool usage...",
                "⚙️ Processing complex logic...",
                "🎯 Refining approach...",
                "✨ Almost there...",
            ]
            while process.returncode is None:
                await asyncio.sleep(5)
                if process.returncode is None:
                    ticks += 1
                    now = asyncio.get_event_loop().time()
                    silence_duration = now - last_activity[0]
                    
                    # Only log if silent for > 5 seconds
                    if silence_duration > 5:
                        # Rotate through phases for variety
                        phase_idx = min(ticks - 1, len(phases) - 1)
                        phase_msg = phases[phase_idx % len(phases)]
                        
                        # Calculate estimated time remaining (rough heuristic)
                        elapsed_mins = int(silence_duration // 60)
                        elapsed_secs = int(silence_duration % 60)
                        time_str = f"{elapsed_mins}m {elapsed_secs}s" if elapsed_mins > 0 else f"{elapsed_secs}s"
                        
                        msg = f"⏳ [VIBE-LIVE] {phase_msg} (Elapsed: {time_str})"
                        logger.info(msg)
                        asyncio.create_task(safe_notify(msg))
                        
                        # Add context message every 30 seconds
                        if ticks % 6 == 0:
                            context_msg = "💡 [VIBE-INFO] Vibe CLI buffers responses - results will appear after processing completes."
                            logger.info(context_msg)
                            asyncio.create_task(safe_notify(context_msg))
                    
                    # Warning after long silence (2+ minutes)
                    if silence_duration > 120 and ticks % 12 == 0:
                        warn_msg = f"⚠️ [VIBE-ALERT] Long processing time ({int(silence_duration)}s). Complex tasks may take several minutes."
                        logger.warning(warn_msg)
                        asyncio.create_task(safe_notify(warn_msg))
        
        hb_task = asyncio.create_task(heartbeat_worker())
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_chunks, "VIBE-OUT"),
                    process.wait()
                ),
                timeout=float(timeout_s)
            )
        except asyncio.TimeoutError:
            try:
                process.terminate()
                await process.wait()
            except:
                pass
            error_msg = f"Vibe CLI timed out after {timeout_s}s"
            logger.error(f"❌ [VIBE-LIVE] {error_msg}")
            return {"success": False, "error": error_msg}
        finally:
            hb_task.cancel()

        stdout = strip_ansi("".join(stdout_chunks))
        stderr = strip_ansi("".join(stderr_chunks))

        logger.info(f"[VIBE] Exit code: {process.returncode}")
        
        if process.returncode != 0 and not stdout:
            return {
                "error": "Vibe CLI returned non-zero exit code",
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "command": argv,
            }

        return {
            "success": True,
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "command": argv,
        }

    except FileNotFoundError:
        error_msg = f"Vibe CLI not found: '{argv[0]}'. Ensure it is installed and on PATH."
        logger.error(f"[VIBE] {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Vibe CLI execution failed: {e}"
        logger.error(f"[VIBE] {error_msg}")
        return {"error": error_msg, "command": argv}

async def _run_vibe_programmatic(
    prompt: str,
    cwd: Optional[str],
    timeout_s: float,
    output_format: str = "json",
    auto_approve: bool = True,
    max_turns: Optional[int] = None,
    max_price: Optional[float] = None,
    resume: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    ctx: Any = None,
) -> Dict[str, Any]:
    """
    Execute Vibe in programmatic mode with -p flag.

    This is the PRIMARY method for interacting with Vibe from MCP.
    Uses CLI mode, NOT interactive TUI, so all output goes to logs.

    Args:
        prompt: The prompt/query to send to Vibe
        cwd: Working directory for execution
        timeout_s: Timeout in seconds
        output_format: 'text', 'json', or 'streaming'
        auto_approve: Auto-approve all tool calls (default True)
        max_turns: Maximum assistant turns
        enabled_tools: List of specific tools to enable
    """
    vibe_path = _resolve_vibe_binary()
    if not vibe_path:
        return {"error": f"Vibe CLI not found on PATH (binary='{VIBE_BINARY}')"}

    prompt_path_to_clean = None

    # Handle long prompts
    final_prompt, prompt_path_to_clean = _prepare_prompt_arg(prompt, cwd)

    try:
        # Build command with programmatic flags
        argv: List[str] = [vibe_path, "-p", final_prompt]

        # Output format for structured responses - use 'streaming' for real-time visibility
        # even if the final result is parsed as JSON
        argv.extend(["--output", "streaming"])

        # Auto-approve for automation
        if auto_approve:
            argv.append("--auto-approve")

        # Max turns limit
        if max_turns:
            argv.extend(["--max-turns", str(max_turns)])

        # Max price limit
        if max_price:
            argv.extend(["--max-price", str(max_price)])

        # Resume session
        if resume:
            argv.extend(["--resume", str(resume)])

        # Specific tools
        if enabled_tools:
            for tool in enabled_tools:
                argv.extend(["--enabled-tools", tool])

        logger.info(f"[VIBE PROGRAMMATIC] Prompt: {prompt[:100]}...")

        result = await _run_vibe(
            argv=argv,
            cwd=cwd,
            timeout_s=timeout_s,
            extra_env=None,
            ctx=ctx,
        )

        # Parse JSON output from stream chunks
        if result.get("success") and result.get("stdout"):
            stdout_str = result.get("stdout", "")
            
            # If we used streaming, the final result is a concatenation of 'content' fields
            if "--output streaming" in " ".join(argv):
                full_content = []
                for line in stdout_str.splitlines():
                    try:
                        data = json.loads(line)
                        if data.get("role") == "assistant" and data.get("content"):
                            full_content.append(data.get("content"))
                    except:
                        continue
                
                final_text = "".join(full_content)
                if final_text.strip():
                    # Attempt to parse the reconstructed text as JSON if that was the original intent
                    if output_format == "json":
                        try:
                            result["parsed_response"] = json.loads(final_text)
                            logger.info("[VIBE] Reconstructed and parsed JSON from stream")
                        except:
                            result["parsed_response"] = final_text
                    else:
                        result["parsed_response"] = final_text
            else:
                # Fallback for standard non-streaming formats
                if output_format == "json":
                    try:
                        result["parsed_response"] = json.loads(stdout_str)
                        logger.info("[VIBE] Parsed JSON response successfully")
                    except:
                        result["parsed_response"] = None

        return result
    finally:
        # Cleanup temporary prompt file
        if prompt_path_to_clean and os.path.exists(prompt_path_to_clean):
            try:
                os.remove(prompt_path_to_clean)
                logger.info(f"[VIBE] Cleaned up prompt file: {prompt_path_to_clean}")
            except Exception as e:
                logger.warning(f"[VIBE] Failed to cleanup prompt file: {e}")


@server.tool()
async def vibe_which() -> Dict[str, Any]:
    """
    Locate the Vibe CLI binary path and version.

    Returns:
        Dict with 'binary' path and 'version' if successful.
    """
    vibe_path = _resolve_vibe_binary()
    if not vibe_path:
        return {"error": f"Vibe CLI not found on PATH (binary='{VIBE_BINARY}')"}

    # Get version
    try:
        logger.info(f"⚡ [VIBE-LIVE] Starting Vibe CLI to get version...")
        process = await asyncio.create_subprocess_exec(
            vibe_path, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        version = stdout.decode().strip() if process.returncode == 0 else "unknown"
    except Exception:
        version = "unknown"

    return {"success": True, "binary": vibe_path, "version": version}


@server.tool()
async def vibe_prompt(
    ctx: Context,
    prompt: str,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
    output_format: str = "json",
    auto_approve: bool = True,
    max_turns: Optional[int] = 10,
    max_price: Optional[float] = None,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a prompt to Vibe AI agent in PROGRAMMATIC mode (CLI, not TUI).

    This is the PRIMARY tool for interacting with Vibe.
    All output is logged and visible in the Electron app.

    Args:
        prompt: The message/query for Vibe AI (Mistral-powered)
        cwd: Working directory for execution
        timeout_s: Timeout in seconds (default 300)
        output_format: Response format - 'text', 'json', or 'streaming' (default 'json')
        auto_approve: Auto-approve tool calls without confirmation (default True)
        max_turns: Maximum conversation turns (default 10)
        max_price: Maximum cost in dollars
        resume: Session ID to continue from

    Returns:
        Dict with 'success', 'stdout', 'parsed_response' (if JSON), 'stderr'

    Example:
        vibe_prompt(
            prompt="Analyze the error in main.py line 45 and fix it",
            cwd="/path/to/project",
            timeout_s=120
        )
    """
    eff_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
    eff_cwd = cwd if cwd is not None else VIBE_WORKSPACE
    
    # Ensure workspace exists
    if not os.path.exists(eff_cwd):
        os.makedirs(eff_cwd, exist_ok=True)

    logger.info(f"[VIBE] Processing prompt: {prompt[:100]}... (CWD: {eff_cwd})")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=eff_cwd,
        timeout_s=eff_timeout,
        output_format=output_format,
        auto_approve=auto_approve,
        max_turns=max_turns,
        max_price=max_price,
        resume=resume,
        ctx=ctx,
    )


@server.tool()
async def vibe_analyze_error(
    ctx: Context,
    error_message: str,
    log_context: Optional[str] = None,
    recovery_history: Optional[str] = None,
    file_path: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
    auto_fix: bool = True,
) -> Dict[str, Any]:
    """
    Deep error analysis and optional auto-fix using Vibe AI.

    This tool is designed for self-healing scenarios when Tetyana
    or Grisha encounter errors they cannot resolve.

    Args:
        error_message: The error message or stack trace to analyze
        log_context: Recent log entries for context
        recovery_history: Summary of past recovery attempts to avoid repeating failures
        file_path: Path to the file with the error (if known)
        cwd: Working directory
        timeout_s: Timeout (default 300s for deep analysis)
        auto_fix: Whether to automatically fix the issue (default True)

    Returns:
        Analysis results with suggested or applied fixes
    """
    # Construct a detailed prompt for error analysis
    prompt_parts = [
        "SYSTEM: You are the Senior Self-Healing Engineer for AtlasTrinity.",
        "ROLE: Analyze and repair the Trinity runtime and its MCP servers.",
        "",
        "DATABASE SCHEMA VISIBILITY:",
        "- 'sessions': id, started_at, ended_at, metadata_blob",
        "- 'tasks': id, session_id, goal, status, created_at, completed_at",
        "- 'task_steps': id, task_id, sequence_number, action, tool, status, error_message, duration_ms",
        "- 'tool_executions': id, step_id, task_id, server_name, tool_name, arguments, result",
        "- 'logs': timestamp, level, source, message",
        "- 'recovery_attempts': step_id, success, vibe_text, error_before",
        "",
        f"CONTEXT:",
        f"- System Root (AtlasTrinity): {SYSTEM_ROOT}",
        f"- Target Directory (Task): {cwd or VIBE_WORKSPACE}",
        f"- Logs Directory: {LOG_DIR}",
        "- OS: macOS",
        "- Internal DB: PostgreSQL (Schema: sessions, tasks, task_steps, tool_executions, logs)",
        "  - Use 'vibe_get_system_context' to find current IDs.",
        "  - Use 'vibe_check_db' to query execution history and past failures.",
        "",
        f"ERROR MESSAGE:\n{error_message}",
    ]

    if log_context:
        prompt_parts.append(f"\nRECENT LOGS:\n{log_context}")

    if recovery_history:
        prompt_parts.append(f"\nRECOVERY HISTORY (PAST ATTEMPTS):\n{recovery_history}\nIMPORTANT: Do NOT repeat strategies that have already failed unless you have a new insight into why they failed.")

    if file_path:
        prompt_parts.append(f"\nFILE PATH: {file_path}")
    else:
        # Try to parse from error message if not provided
        trace_info = _parse_stack_trace(error_message)
        if trace_info:
            file_path = trace_info["file"]
            prompt_parts.append(f"\nDETECTED FILE FROM TRACE: {file_path} (Line {trace_info['line']})")

    # Enhance with file content if available
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Context limit for clarity
                if len(content) > 12000:
                    content = content[:12000] + "\n... [TRUNCATED FILE CONTENT] ..."
                prompt_parts.append(f"\nTARGET FILE CONTENT ({file_path}):\n```python\n{content}\n```")
        except Exception as e:
            prompt_parts.append(f"\n(Could not read file {file_path}: {e})")


    if auto_fix:
        prompt_parts.extend(
            [
                "",
                "INSTRUCTIONS:",
                "1. ROOT CAUSE ANALYSIS (RCA): Before fixing, perform a deep analysis of WHY this error occurred. Use 'grep' or 'read_file' to investigate dependencies and state.",
                "2. Check the 'RECOVERY HISTORY' above. If past attempts failed, analyze the failures and choose a DIFFERENT approach.",
                "3. Use 'vibe_code_search' or 'vibe_check_db' if needed to understand the environment.",
                "4. ACTIVELY FIX the issue (edit code, run commands).",
                "5. If you modify Swift code in 'vendor/mcp-server-macos-use', you MUST recompile it by running 'swift build -c release' in that directory.",
                "6. After any fix to an MCP server, use 'vibe_restart_mcp_server(server_name)' to apply changes.",
                "7. Verify the fix works by running tests or the failing command again.",
                "8. Provide a detailed summary including the Root Cause found.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "INSTRUCTIONS:",
                "1. Analyze the error thoroughly",
                "2. Identify the root cause",
                "3. Suggest specific fixes (without applying them)",
                "4. Explain why each fix would work",
            ]
        )

    prompt = "\n".join(prompt_parts)
    eff_timeout = timeout_s if timeout_s is not None else 300.0
    eff_cwd = cwd if cwd is not None else VIBE_WORKSPACE
    
    # Ensure workspace exists
    if not os.path.exists(eff_cwd):
        os.makedirs(eff_cwd, exist_ok=True)

    logger.info(f"[VIBE] Starting error analysis (auto_fix={auto_fix}, CWD={eff_cwd})")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=eff_cwd,
        timeout_s=eff_timeout,
        output_format="json",
        auto_approve=auto_fix,  # Only auto-approve if auto_fix is True
        max_turns=15,  # More turns for complex debugging
        ctx=ctx,
    )


@server.tool()
async def vibe_implement_feature(
    ctx: Context,
    goal: str,
    context_files: List[str] = [],
    constraints: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = 1200,
) -> Dict[str, Any]:
    """
    "Deep Coding" mode: Implements a complex feature or refactoring.
    
    Acts as a Senior Software Architect to plan, implement, and verify changes across multiple files.
    
    Args:
        goal: The high-level objective (e.g. "Add a user profile page with API and DB support")
        context_files: List of file paths relevant to the task
        constraints: Technical constraints or architectural guidelines
        cwd: Working directory
        timeout_s: Timeout in seconds (default 1200s for deep work)
        
    Returns:
        Implementation report with changed files and verification status.
    """
    
    # 1. Gather Context Content
    file_contents = []
    for fpath in context_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                    if len(content) > 12000:
                         content = content[:12000] + "\n... [TRUNCATED] ..."
                    file_contents.append(f"File: {fpath}\n```\n{content}\n```")
            except Exception as e:
                file_contents.append(f"File: {fpath} (Error reading: {e})")
        else:
            file_contents.append(f"File: {fpath} (Not found, will create)")
            
    context_str = "\n\n".join(file_contents)
    
    # 2. Construct Prompt for "Architect" Persona
    prompt = f"""
SYSTEM: You are the Senior Software Architect and Lead Developer for AtlasTrinity.
ROLE: Implement a complex feature efficiently and robustly.

GOAL: {goal}

CONTEXT FILES:
{context_str}

CONSTRAINTS:
{constraints or "Standard project guidelines apply."}
- System Root (AtlasTrinity): {SYSTEM_ROOT}
- Project Root (Current Work): {cwd or VIBE_WORKSPACE}

INSTRUCTIONS:
1. PLAN: Analyze the goal and files. enhancing existing architecture.
2. IMPLEMENT: 
   - Write/Edit the necessary code using file operations.
   - Handle imports and dependencies.
   - You can edit multiple files.
3. VERIFY:
   - Run simple checks or "python -m" to ensure no syntax errors.
   - If changes involve UI, describe how they should look.
4. REPORT:
   - Report exactly which files were modified.
   - Confirm success or explain limitations.

EXECUTE NOW. ALLOWED 30 TURNS.
"""

    eff_cwd = cwd if cwd is not None else VIBE_WORKSPACE
    if not os.path.exists(eff_cwd):
        os.makedirs(eff_cwd, exist_ok=True)
        
    logger.info(f"[VIBE-DEEP] Starting Implementation: {goal[:100]}...")
    
    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=eff_cwd,
        timeout_s=timeout_s or 1200,
        output_format="json",
        auto_approve=True, # Deep coding requires autonomy
        max_turns=30,      # High turn limit for complex tasks
        ctx=ctx
    )


@server.tool()
async def vibe_code_review(
    ctx: Context,
    file_path: str,
    focus_areas: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Request a code review from Vibe AI for a specific file.

    Args:
        file_path: Path to the file to review
        focus_areas: Optional specific areas to focus on (e.g., "security", "performance")
        cwd: Working directory
        timeout_s: Timeout in seconds

    Returns:
        Code review analysis with suggestions
    """
    prompt_parts = [
        f"CODE REVIEW REQUEST: {file_path}",
        "",
        "Please review this code and provide:",
        "1. Overall code quality assessment",
        "2. Potential bugs or issues",
        "3. Security concerns (if any)",
        "4. Performance improvements",
        "5. Code style and best practices",
    ]

    if focus_areas:
        prompt_parts.append(f"\nFOCUS AREAS: {focus_areas}")

    prompt = "\n".join(prompt_parts)
    eff_timeout = timeout_s if timeout_s is not None else 120.0

    logger.info(f"[VIBE] Starting code review for: {file_path}")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=cwd,
        timeout_s=eff_timeout,
        output_format="json",
        auto_approve=False,  # Read-only mode for reviews
        max_turns=5,
        ctx=ctx,
    )


@server.tool()
async def vibe_smart_plan(
    ctx: Context,
    objective: str,
    context: Optional[str] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate a smart execution plan for a complex objective.

    Uses Vibe AI to create a structured plan with steps.

    Args:
        objective: The goal or task to plan for
        context: Additional context (existing code, constraints, etc.)
        cwd: Working directory
        timeout_s: Timeout in seconds

    Returns:
        Structured plan with steps
    """
    prompt_parts = [
        "SMART PLANNING REQUEST",
        "",
        f"OBJECTIVE: {objective}",
    ]

    if context:
        prompt_parts.append(f"\nCONTEXT:\n{context}")

    prompt_parts.extend(
        [
            "",
            "Create a detailed, step-by-step execution plan.",
            "For each step, specify:",
            "- Action to perform",
            "- Required tools/commands",
            "- Expected outcome",
            "- Verification criteria",
        ]
    )

    prompt = "\n".join(prompt_parts)
    eff_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

    logger.info(f"[VIBE] Generating smart plan for: {objective[:50]}...")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=cwd,
        timeout_s=eff_timeout,
        output_format="json",
        auto_approve=False,  # Planning mode, no actions
        max_turns=3,
        ctx=ctx,
    )


@server.tool()
async def vibe_execute_subcommand(
    ctx: Context,
    subcommand: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Execute a specific Vibe CLI subcommand (for non-AI operations).

    This is for utility commands like list-editors, run cleanup, etc.
    For AI interactions, use vibe_prompt instead.

    Args:
        subcommand: The vibe subcommand (e.g., 'list-editors', 'run')
        args: Optional arguments for the subcommand
        cwd: Working directory
        timeout_s: Timeout in seconds
        env: Additional environment variables

    Allowed subcommands:
        list-editors, list-modules, run, enable, disable, install,
        agent-reset, agent-on, agent-off, vibe-status, vibe-continue,
        vibe-cancel, vibe-help, eternal-engine, screenshots

    Blocked (use vibe_prompt instead):
        tui, agent-chat, self-healing-status, self-healing-scan
    """
    vibe_path = _resolve_vibe_binary()
    if not vibe_path:
        return {"error": f"Vibe CLI not found on PATH (binary='{VIBE_BINARY}')"}

    sub = (subcommand or "").strip()
    if not sub:
        return {"error": "Missing subcommand"}

    if sub in BLOCKED_SUBCOMMANDS:
        return {
            "error": f"Subcommand '{sub}' is interactive/TUI mode and blocked. Use vibe_prompt() for AI interactions.",
            "suggestion": "Use vibe_prompt(prompt='your query') instead for programmatic AI access.",
        }

    if sub not in ALLOWED_SUBCOMMANDS:
        return {
            "error": f"Subcommand not recognized: '{sub}'.",
            "allowed": sorted(ALLOWED_SUBCOMMANDS),
        }

    argv: List[str] = [vibe_path, sub]
    if args:
        argv.extend([str(a) for a in args])

    eff_timeout = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S

    return await _run_vibe(argv=argv, cwd=cwd, timeout_s=eff_timeout, extra_env=env, ctx=ctx)


@server.tool()
async def vibe_ask(
    ctx: Context,
    question: str,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ask Vibe AI a quick question (read-only, no tool execution).

    Similar to vibe_prompt but with --plan flag for read-only mode.

    Args:
        question: The question to ask
        cwd: Working directory
        timeout_s: Timeout in seconds

    Returns:
        AI response without any file modifications
    """
    vibe_path = _resolve_vibe_binary()
    if not vibe_path:
        return {"error": f"Vibe CLI not found on PATH (binary='{VIBE_BINARY}')"}

    # Use helper to handle large prompts
    final_question, prompt_path_to_clean = _prepare_prompt_arg(question, cwd)

    argv = [vibe_path, "-p", final_question, "--output", "json", "--plan"]

    eff_timeout = timeout_s if timeout_s is not None else 120.0  # 2 minutes for warmup

    logger.info(f"[VIBE] Asking question: {question[:50]}...")

    try:
        result = await _run_vibe(argv=argv, cwd=cwd, timeout_s=eff_timeout, extra_env=None, ctx=ctx)

        # Parse JSON if possible
        if result.get("success") and result.get("stdout"):
            try:
                result["parsed_response"] = json.loads(result["stdout"])
            except json.JSONDecodeError:
                result["parsed_response"] = None

        return result
    finally:
        # Cleanup temporary prompt file
        if prompt_path_to_clean and os.path.exists(prompt_path_to_clean):
            try:
                os.remove(prompt_path_to_clean)
                logger.info(f"[VIBE] Cleaned up prompt file: {prompt_path_to_clean}")
            except Exception as e:
                logger.warning(f"[VIBE] Failed to cleanup prompt file: {e}")


@server.tool()
async def vibe_list_sessions(limit: int = 10) -> Dict[str, Any]:
    """
    List recent Vibe session logs with token usage and metrics.
    Useful for tracking costs, context size, and session IDs for resuming.
    """
    if not VIBE_SESSION_DIR.exists():
        return {"error": f"Session logs directory not found at {VIBE_SESSION_DIR}"}
    
    # Get all json files in the session directory
    files = list(VIBE_SESSION_DIR.glob("session_*.json"))
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    sessions = []
    for f in files[:limit]:
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
                meta = data.get("metadata", {})
                stats = meta.get("stats", {})
                
                # Extract some preview content
                messages = data.get("messages", [])
                preview = ""
                if messages:
                    # Find first user message for a better summary
                    for msg in messages:
                        if msg.get("role") == "user":
                            preview = msg.get("content", "")[:100]
                            break
                
                sessions.append({
                    "session_id": meta.get("session_id"),
                    "timestamp": meta.get("start_time"),
                    "working_directory": meta.get("environment", {}).get("working_directory", "unknown"),
                    "steps": stats.get("steps", 0),
                    "prompt_tokens": stats.get("session_prompt_tokens", 0),
                    "completion_tokens": stats.get("session_completion_tokens", 0),
                    "summary": preview,
                    "file_name": f.name
                })
        except Exception as e:
            logger.debug(f"Failed to parse session file {f.name}: {e}")
            continue
    
    return {"sessions": sessions, "count": len(sessions)}


@server.tool()
async def vibe_session_details(session_id_or_file: str) -> Dict[str, Any]:
    """
    Get full details of a specific Vibe session including history and exact token counts.
    """
    target_path = None
    
    # Check if it's an absolute path
    if os.path.isabs(session_id_or_file) and os.path.exists(session_id_or_file):
        target_path = Path(session_id_or_file)
    # Check if it's just a filename in the session dir
    elif (VIBE_SESSION_DIR / session_id_or_file).exists():
        target_path = VIBE_SESSION_DIR / session_id_or_file
    # Search by session_id
    else:
        # Try to find file containing the session_id
        files = list(VIBE_SESSION_DIR.glob(f"*session*{session_id_or_file}*.json"))
        if files:
            target_path = files[0]
            
    if not target_path:
        return {"error": f"Session '{session_id_or_file}' not found."}
        
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to read session details: {str(e)}"}


@server.tool()
async def vibe_check_db(query: str) -> Dict[str, Any]:
    """
    Execute a read-only SQL query against the AtlasTrinity PostgreSQL database.
    Use this to inspect task execution history, tool results, and system state.
    
    SCHEMA SUMMARY:
    - sessions: (id, started_at)
    - tasks: (id, session_id, goal, status)
    - task_steps: (id, task_id, sequence_number, action, tool, status, error_message)
    - tool_executions: (id, step_id, task_id, server_name, tool_name, arguments, result)
    - logs: (timestamp, level, source, message)
    - recovery_attempts: (step_id, success, vibe_text, error_before)
    
    Args:
        query: SQL SELECT query (e.g. "SELECT * FROM logs ORDER BY timestamp DESC LIMIT 10")
    """
    import asyncpg
    
    # Check for mutative keywords to prevent accidental damage
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER", "GRANT", "REVOKE"]
    if any(f in query.upper() for f in forbidden):
        return {"error": "Only SELECT queries are allowed for safety reasons."}
    
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            rows = await conn.fetch(query)
            # Convert Records to list of dicts for JSON serialization
            result = [dict(r) for r in rows]
            return {"success": True, "count": len(result), "data": result}
        finally:
            await conn.close()
    except Exception as e:
        logger.error(f"[VIBE_DB] Query failed: {e}")
        return {"error": f"Database query failed: {str(e)}", "query": query}


@server.tool()
async def vibe_get_system_context() -> Dict[str, Any]:
    """
    Retrieve the current operational context from the DB (Current Session, Latest Tasks, Last 5 Errors).
    Helps Vibe focus on the 'current picture' before performing deep analysis.
    """
    import asyncpg
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            # 1. Get latest session
            session_row = await conn.fetchrow("SELECT id, started_at FROM sessions ORDER BY started_at DESC LIMIT 1")
            session_id = str(session_row['id']) if session_row else "None"
            
            # 2. Get latest tasks for this session
            tasks = await conn.fetch(
                "SELECT id, goal, status, created_at FROM tasks WHERE session_id = $1 ORDER BY created_at DESC LIMIT 3", 
                session_row['id'] if session_row else None
            )
            
            # 3. Get last 5 system errors
            errors = await conn.fetch("SELECT timestamp, source, message FROM logs WHERE level IN ('ERROR', 'WARNING') ORDER BY timestamp DESC LIMIT 5")
            
            return {
                "success": True,
                "current_session_id": session_id,
                "last_tasks": [dict(t) for t in tasks],
                "recent_errors": [dict(e) for e in errors],
                "system_root": SYSTEM_ROOT,
                "project_root": VIBE_WORKSPACE
            }
        finally:
            await conn.close()
    except Exception as e:
        return {"error": f"Failed to get system context: {e}"}


if __name__ == "__main__":
    server.run()
