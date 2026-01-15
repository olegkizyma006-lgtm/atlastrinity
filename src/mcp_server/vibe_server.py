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

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

# Setup logging for visibility in Electron app
logger = logging.getLogger("vibe_mcp")
logger.setLevel(logging.INFO)

try:
    from .config_loader import get_config_value

    VIBE_BINARY = get_config_value("vibe", "binary", "vibe")
    DEFAULT_TIMEOUT_S = float(get_config_value("vibe", "timeout_s", 300))
    # Increased for large log analysis
    MAX_OUTPUT_CHARS = int(get_config_value("vibe", "max_output_chars", 500000))
    DISALLOW_INTERACTIVE = bool(get_config_value("vibe", "disallow_interactive", True))
except Exception:
    VIBE_BINARY = "vibe"
    DEFAULT_TIMEOUT_S = 300.0
    MAX_OUTPUT_CHARS = 500000  # 500KB for large logs
    DISALLOW_INTERACTIVE = True

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


async def _run_vibe(
    argv: List[str],
    cwd: Optional[str],
    timeout_s: float,
    extra_env: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Execute Vibe CLI command and return structured result with real-time logging."""
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    logger.info(f"[VIBE] Executing: {' '.join(argv)}")

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
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode().rstrip()
                chunks.append(text + "\n")
                # Log in real-time for UI visibility
                if "error" in prefix.lower() or "stderr" in prefix.lower():
                    logger.warning(f"[{prefix}] {text}")
                else:
                    # Filter out some common vibe noise if needed, or just show all
                    logger.info(f"[{prefix}] {text}")

        # Run reading tasks concurrently with a timeout
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
            logger.error(f"[VIBE] {error_msg}")
            return {"error": error_msg, "command": argv}

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
    enabled_tools: Optional[List[str]] = None,
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

    # Build command with programmatic flags
    argv: List[str] = [vibe_path, "-p", prompt]

    # Output format for structured responses
    argv.extend(["--output", output_format])

    # Auto-approve for automation
    if auto_approve:
        argv.append("--auto-approve")

    # Max turns limit
    if max_turns:
        argv.extend(["--max-turns", str(max_turns)])

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
    )

    # Parse JSON output if requested
    if output_format == "json" and result.get("success") and result.get("stdout"):
        try:
            parsed = json.loads(result["stdout"])
            result["parsed_response"] = parsed
            logger.info("[VIBE] Parsed JSON response successfully")
        except json.JSONDecodeError:
            # If not valid JSON, keep raw stdout
            logger.warning("[VIBE] Output was not valid JSON, keeping raw text")
            result["parsed_response"] = None

    return result


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
    prompt: str,
    cwd: Optional[str] = None,
    timeout_s: Optional[float] = None,
    output_format: str = "json",
    auto_approve: bool = True,
    max_turns: Optional[int] = 10,
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

    logger.info(f"[VIBE] Processing prompt: {prompt[:100]}...")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=cwd,
        timeout_s=eff_timeout,
        output_format=output_format,
        auto_approve=auto_approve,
        max_turns=max_turns,
    )


@server.tool()
async def vibe_analyze_error(
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
        "AUTONOMOUS ERROR ANALYSIS AND REPAIR",
        "",
        f"CONTEXT:",
        f"- Project Root: {PROJECT_ROOT}",
        f"- Logs Directory: {LOG_DIR}",
        f"- OS: macOS",
        "",
        f"ERROR MESSAGE:\n{error_message}",
    ]

    if log_context:
        prompt_parts.append(f"\nRECENT LOGS:\n{log_context}")

    if file_path:
        prompt_parts.append(f"\nFILE PATH: {file_path}")

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

    logger.info(f"[VIBE] Starting error analysis (auto_fix={auto_fix})")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=cwd,
        timeout_s=eff_timeout,
        output_format="json",
        auto_approve=auto_fix,  # Only auto-approve if auto_fix is True
        max_turns=15,  # More turns for complex debugging
    )


@server.tool()
async def vibe_code_review(
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
    )


@server.tool()
async def vibe_smart_plan(
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
    eff_timeout = timeout_s if timeout_s is not None else 60.0

    logger.info(f"[VIBE] Generating smart plan for: {objective[:50]}...")

    return await _run_vibe_programmatic(
        prompt=prompt,
        cwd=cwd,
        timeout_s=eff_timeout,
        output_format="json",
        auto_approve=False,  # Planning mode, no actions
        max_turns=3,
    )


@server.tool()
async def vibe_execute_subcommand(
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

    return await _run_vibe(argv=argv, cwd=cwd, timeout_s=eff_timeout, extra_env=env)


@server.tool()
async def vibe_ask(
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

    argv = [vibe_path, "-p", question, "--output", "json", "--plan"]

    eff_timeout = timeout_s if timeout_s is not None else 120.0  # 2 minutes for warmup

    logger.info(f"[VIBE] Asking question: {question[:50]}...")

    result = await _run_vibe(argv=argv, cwd=cwd, timeout_s=eff_timeout, extra_env=None)

    # Parse JSON if possible
    if result.get("success") and result.get("stdout"):
        try:
            result["parsed_response"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            result["parsed_response"] = None

    return result


@server.tool()
async def vibe_restart_mcp_server(server_name: str) -> Dict[str, Any]:
    """
    Restart a specific MCP server (e.g., 'macos-use', 'filesystem').
    Use this after modifying source code or fixing a crash.
    """
    from ..brain.mcp_manager import mcp_manager
    success = await mcp_manager.restart_server(server_name)
    return {"success": success, "restarted": server_name}


@server.tool()
async def vibe_query_db(query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Query the internal AtlasTrinity database (PostgreSQL).
    Tables: tasks, task_steps, tool_executions, logs, kg_nodes, kg_edges.
    """
    from ..brain.mcp_manager import mcp_manager
    results = await mcp_manager.query_db(query, params)
    return {"success": True, "results": results}


if __name__ == "__main__":
    server.run()
