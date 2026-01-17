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
import uuid
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from mcp.server.fastmcp import FastMCP, Context

# Setup logging for visibility in Electron app
logger = logging.getLogger("vibe_mcp")
logger.setLevel(logging.INFO)

# Standard path for AtlasTrinity logs
try:
    _log_dir = Path.home() / ".config" / "atlastrinity" / "logs"
    _log_dir.mkdir(parents=True, exist_ok=True)
    # Use simple FileHandler for brain.log visibility
    fh = logging.FileHandler(_log_dir / "brain.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - vibe_mcp - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    # Also keep stderr for internal mcp console visibility
    import sys
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[VIBE_MCP] %(message)s"))
    logger.addHandler(sh)
except Exception:
    pass # Fallback to default if log dir unreachable

try:
    from .config_loader import get_config_value

    VIBE_BINARY = get_config_value("vibe", "binary", "vibe")
    DEFAULT_TIMEOUT_S = float(get_config_value("vibe", "timeout_s", 1200))
    # Increased for large log analysis
    MAX_OUTPUT_CHARS = int(get_config_value("vibe", "max_output_chars", 500000))
    DISALLOW_INTERACTIVE = bool(get_config_value("vibe", "disallow_interactive", True))
    
    # Resolve global vibe_workspace
    VIBE_WORKSPACE = get_config_value("vibe", "workspace", str(Path.home() / ".config" / "atlastrinity" / "vibe_workspace"))
except Exception:
    VIBE_BINARY = "vibe"
    DEFAULT_TIMEOUT_S = 1200.0
    MAX_OUTPUT_CHARS = 500000  # 500KB for large logs
    DISALLOW_INTERACTIVE = True
    VIBE_WORKSPACE = str(Path.home() / ".config" / "atlastrinity" / "vibe_workspace")

from pathlib import Path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
LOG_DIR = str(Path.home() / ".config" / "atlastrinity" / "logs")


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

# Subcommands that are BLOCKED (interactive TUI)
BLOCKED_SUBCOMMANDS = {
    "tui",
    "agent-chat",  # Use vibe_prompt instead for programmatic mode
    "self-healing-status",  # TUI mode, use vibe_prompt for queries
    "self-healing-scan",  # TUI mode, use vibe_prompt for queries
}


server = FastMCP("vibe")


def _truncate(text: str) -> str:
    """Truncate text to max output chars with indicator."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... [TRUNCATED - Output exceeded 500KB] ..."


def _resolve_vibe_binary() -> Optional[str]:
    """Resolve the path to the Vibe CLI binary."""
    if os.path.isabs(VIBE_BINARY) and os.path.exists(VIBE_BINARY):
        return VIBE_BINARY
    return shutil.which(VIBE_BINARY)


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
    """
    if len(prompt) <= 2000:
        return prompt, None

    try:
        eff_cwd = cwd or VIBE_WORKSPACE
        if not os.path.exists(eff_cwd):
            os.makedirs(eff_cwd, exist_ok=True)
            
        prompt_file = f"vibe_instructions_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:6]}.md"
        prompt_path = os.path.join(eff_cwd, prompt_file)
        
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write("# INSTRUCTIONS FOR VIBE AGENT\n\n")
            f.write(prompt)
        
        logger.info(f"[VIBE] Large prompt ({len(prompt)} chars) offloaded to {prompt_path}")
        return f"Please read and execute the instructions detailed in the file: {prompt_file}", prompt_path

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

    msg_start = f"⚡ [VIBE-LIVE] Initializing... (Timeout: {timeout_s}s)"
    logger.info(msg_start)
    if ctx: asyncio.create_task(ctx.info(msg_start))

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout_chunks = []
        stderr_chunks = []

        async def read_stream(stream, chunks, prefix):
            buffer = ""
            while True:
                # Read chunks instead of lines for more immediate feedback
                data = await stream.read(512)
                if not data:
                    break
                
                text_chunk = data.decode(errors='replace')
                chunks.append(text_chunk)
                buffer += text_chunk
                
                lines = []
                if "\n" in buffer:
                    lines = buffer.split("\n")
                    # Last element might be incomplete line, keep it
                    buffer = lines.pop()
                    
                for raw_text in lines:
                    if not raw_text.strip():
                        continue
                    
                    # Robust JSON detection
                    stripped = raw_text.strip()
                    is_json = False
                    
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            # Try to parse as JSON first (vibe --output streaming / json)
                            data_json = json.loads(stripped)
                            is_json = True
                            
                            thoughts = data_json.get("reasoning_content") or data_json.get("thought") or ""
                            content = data_json.get("content") or ""
                            tool_calls = data_json.get("tool_calls")
                            role = data_json.get("role")
                            
                            if thoughts:
                                # Truncate very long reasoning
                                snippet = thoughts[:500] + ("..." if len(thoughts) > 500 else "")
                                msg = f"🧠 [VIBE-THOUGHT] {snippet}"
                                logger.info(msg)
                                if ctx: asyncio.create_task(ctx.info(msg))
                            
                            elif tool_calls:
                                # Vibe often emits multiple tool calls in streaming
                                for tc in tool_calls:
                                    func = tc.get("function", {})
                                    f_name = func.get("name", "unknown_tool")
                                    f_args = func.get("arguments", "{}")
                                    # Show a bit of the arguments for context
                                    args_snippet = f_args[:100] + ("..." if len(f_args) > 100 else "")
                                    msg = f"🛠️ [VIBE-ACTION] Using tool: {f_name} | Args: {args_snippet}"
                                    logger.info(msg)
                                    if ctx: asyncio.create_task(ctx.info(msg))
                            
                            elif content:
                                # The actual AI response snippets
                                snippet = content.strip().replace("\n", " ")[:300]
                                if snippet:
                                    msg = f"📝 [VIBE-GEN] {snippet}..."
                                    logger.info(msg)
                                    if ctx: asyncio.create_task(ctx.info(msg))
                            
                            elif role == "user":
                                msg = f"👤 [VIBE-USER] {data_json.get('content', '')[:200]}..."
                                logger.info(msg)
                                
                        except Exception:
                            is_json = False # Fall back to raw text logging

                    # Raw text logging (for non-JSON parts or parsing failures)
                    if not is_json:
                        # Extract "milestones" from raw CLI output
                        lower_text = stripped.lower()
                        if any(kw in lower_text for kw in ["creating", "writing", "saved", "modified", "editing"]):
                            msg = f"📂 [VIBE-FILES] {stripped}"
                            logger.info(msg)
                            if ctx: asyncio.create_task(ctx.info(msg))
                        elif "step" in lower_text and ":" in lower_text:
                            msg = f"📍 [VIBE-STEP] {stripped}"
                            logger.info(msg)
                            if ctx: asyncio.create_task(ctx.info(msg))
                        elif "error" in lower_text or "fail" in lower_text:
                            msg = f"⚠️ [VIBE-ALERT] {stripped}"
                            logger.warning(msg)
                            if ctx: asyncio.create_task(ctx.error(msg))
                        else:
                            # General output - log to file, but only small snippets to UI to avoid spam
                            msg = f"📺 [VIBE-LIVE] {stripped[:200]}{'...' if len(stripped) > 200 else ''}"
                            logger.info(msg)
                            if ctx and len(stripped) < 500:
                                asyncio.create_task(ctx.info(msg))

        # Run reading tasks concurrently with a timeout
        # Heartbeat task for deep reasoning phase
        async def heartbeat_worker():
            reasoning_ticks = 0
            while process.returncode is None:
                await asyncio.sleep(45)
                if process.returncode is None:
                    reasoning_ticks += 1
                    msg = f"🧠 [VIBE-LIVE] Vibe is deep-reasoning... (Tick {reasoning_ticks}, API response pending)"
                    logger.info(msg)
                    if ctx: asyncio.create_task(ctx.info(msg))
        
        hb_task = asyncio.create_task(heartbeat_worker())
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_chunks, "VIBE-OUT"),
                    read_stream(process.stderr, stderr_chunks, "VIBE-ERR"),
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

        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)

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
        f"CONTEXT:",
        f"- Project Root: {PROJECT_ROOT}",
        f"- Logs Directory: {LOG_DIR}",
        "- OS: macOS",
        "- Internal DB: PostgreSQL (Schema: sessions, tasks, task_steps, tool_executions, logs)",
        "  - 'tool_executions' table contains RAW results of all agent actions.",
        "",
        f"ERROR MESSAGE:\n{error_message}",
    ]

    if log_context:
        prompt_parts.append(f"\nRECENT LOGS:\n{log_context}")

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
                "1. Analyze the error thoroughly using logs and source code.",
                "2. Identify the root cause.",
                "3. ACTIVELY FIX the issue (edit code, run commands).",
                "4. If you modify Swift code in 'vendor/mcp-server-macos-use', you MUST recompile it by running 'swift build -c release' in that directory.",
                "5. After any fix to an MCP server, use 'vibe_restart_mcp_server(server_name)' to apply changes.",
                "6. Verify the fix works.",
                "7. Provide a detailed summary.",
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
    session_dir = Path.home() / ".vibe" / "logs" / "session"
    if not session_dir.exists():
        return {"error": "Session logs directory not found"}
    
    # Get all json files in the session directory
    files = list(session_dir.glob("session_*.json"))
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
    session_dir = Path.home() / ".vibe" / "logs" / "session"
    target_path = None
    
    # Check if it's an absolute path
    if os.path.isabs(session_id_or_file) and os.path.exists(session_id_or_file):
        target_path = Path(session_id_or_file)
    # Check if it's just a filename in the session dir
    elif (session_dir / session_id_or_file).exists():
        target_path = session_dir / session_id_or_file
    # Search by session_id
    else:
        # Try to find file containing the session_id
        files = list(session_dir.glob(f"*session*{session_id_or_file}*.json"))
        if files:
            target_path = files[0]
            
    if not target_path:
        return {"error": f"Session '{session_id_or_file}' not found."}
        
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to read session details: {str(e)}"}


if __name__ == "__main__":
    server.run()
