"""
Vibe MCP Server - Hyper-Refactored Implementation

This server wraps the Mistral Vibe CLI in MCP-compliant programmatic mode.
Fully aligned with official Mistral Vibe documentation and configuration.

Key Features:
- Full configuration support (providers, models, agents, tool permissions)
- 17 MCP tools covering all Vibe capabilities
- Streaming output with real-time notifications
- Proper error handling and resource cleanup
- Session persistence and resumption
- Dynamic model/provider switching

Based on official Mistral Vibe documentation:
https://docs.mistral.ai/vibe/configuration/

Author: AtlasTrinity Team
Date: 2026-01-20
Version: 3.0 (Hyper-Refactored)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from re import Pattern
from typing import Any, Literal

from mcp.server import FastMCP
from mcp.server.fastmcp import Context

from .vibe_config import (
    AgentMode,
    ProviderConfig,
    VibeConfig,
)

# =============================================================================
# SETUP: Logging, Configuration, Constants
# =============================================================================

logger = logging.getLogger("vibe_mcp")
logger.setLevel(logging.DEBUG)

# Setup file and stream handlers
try:
    config_root = Path.home() / ".config" / "atlastrinity"
    log_dir = config_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler
    fh = logging.FileHandler(log_dir / "vibe_server.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
except Exception as e:
    print(f"[VIBE] Warning: Could not setup file logging: {e}")

sh = logging.StreamHandler(sys.stderr)
sh.setLevel(logging.INFO)
sh.setFormatter(logging.Formatter("[VIBE_MCP] %(levelname)s: %(message)s"))
logger.addHandler(sh)

# Load configuration
try:
    from .config_loader import CONFIG_ROOT, PROJECT_ROOT, get_config_value

    VIBE_BINARY = get_config_value("mcp.vibe", "binary", "vibe")
    DEFAULT_TIMEOUT_S = float(get_config_value("mcp.vibe", "timeout_s", 600))
    MAX_OUTPUT_CHARS = int(get_config_value("mcp.vibe", "max_output_chars", 500000))
    VIBE_WORKSPACE = get_config_value("mcp.vibe", "workspace", str(CONFIG_ROOT / "vibe_workspace"))
    VIBE_CONFIG_FILE = get_config_value("mcp.vibe", "config_file", None)
    AGENT_MODEL_OVERRIDE = get_config_value("agents.tetyana", "model", None)

    if not AGENT_MODEL_OVERRIDE:
        logger.warning(
            "[VIBE] AGENT_MODEL_OVERRIDE not set in config, strict configuration enforced"
        )

except Exception:

    def get_config_value(section: str, key: str, default: Any) -> Any:
        return default

    VIBE_BINARY = "vibe"
    DEFAULT_TIMEOUT_S = 600.0
    MAX_OUTPUT_CHARS = 500000
    CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    VIBE_WORKSPACE = str(CONFIG_ROOT / "vibe_workspace")
    VIBE_CONFIG_FILE = None

# Derived paths
SYSTEM_ROOT = str(PROJECT_ROOT)
LOG_DIR = str(CONFIG_ROOT / "logs")
INSTRUCTIONS_DIR = str(Path(VIBE_WORKSPACE) / "instructions")
VIBE_SESSION_DIR = Path.home() / ".vibe" / "logs" / "session"
DATABASE_URL = get_config_value(
    "database", "url", f"sqlite+aiosqlite:///{CONFIG_ROOT}/atlastrinity.db"
)

# ANSI escape code pattern for stripping colors
ANSI_ESCAPE: Pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Allowed subcommands (CLI-only, no TUI)
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

# Blocked subcommands (interactive TUI)
BLOCKED_SUBCOMMANDS = {"tui", "agent-chat", "self-healing-status", "self-healing-scan"}

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Vibe configuration (loaded at startup)
_vibe_config: VibeConfig | None = None
_current_mode: AgentMode = AgentMode.AUTO_APPROVE
_current_model: str | None = None


def get_vibe_config() -> VibeConfig:
    """Get or load the Vibe configuration."""
    global _vibe_config
    if _vibe_config is None:
        config_path = Path(VIBE_CONFIG_FILE) if VIBE_CONFIG_FILE else None
        _vibe_config = VibeConfig.load(config_path=config_path)
        logger.info(f"[VIBE] Loaded configuration: active_model={_vibe_config.active_model}")
    return _vibe_config


def reload_vibe_config() -> VibeConfig:
    """Force reload the Vibe configuration."""
    global _vibe_config
    _vibe_config = None
    return get_vibe_config()


# =============================================================================
# INITIALIZATION
# =============================================================================

server = FastMCP("vibe")

logger.info(
    f"[VIBE] Server initialized | "
    f"Binary: {VIBE_BINARY} | "
    f"Workspace: {VIBE_WORKSPACE} | "
    f"Timeout: {DEFAULT_TIMEOUT_S}s"
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    if not isinstance(text, str):
        return str(text)
    return ANSI_ESCAPE.sub("", text)


async def is_network_available(
    host: str = "api.mistral.ai", port: int = 443, timeout: float = 3.0
) -> bool:
    """Check if the network and specific host are reachable."""
    try:
        await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
        return True
    except (TimeoutError, OSError) as e:
        logger.warning(f"[VIBE] Network check failed for {host}:{port}: {e}")
        return False


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Truncate text with indicator if exceeded."""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[TRUNCATED: Output exceeded {max_chars} chars]..."


def resolve_vibe_binary() -> str | None:
    """Resolve the path to the Vibe CLI binary."""
    # Try ~/.local/bin first (common location)
    local_bin = os.path.expanduser("~/.local/bin/vibe")
    if os.path.exists(local_bin):
        return local_bin

    # Try absolute path from config
    if os.path.isabs(VIBE_BINARY) and os.path.exists(VIBE_BINARY):
        return VIBE_BINARY

    # Search PATH
    found = shutil.which(VIBE_BINARY)
    if found:
        return found

    logger.warning(f"Vibe binary '{VIBE_BINARY}' not found")
    return None


def prepare_workspace_and_instructions() -> None:
    """Ensure necessary directories exist."""
    try:
        Path(VIBE_WORKSPACE).mkdir(parents=True, exist_ok=True)
        Path(INSTRUCTIONS_DIR).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Workspace ready: {VIBE_WORKSPACE}")
    except Exception as e:
        logger.error(f"Failed to create workspace: {e}")


def cleanup_old_instructions(max_age_hours: int = 24) -> int:
    """Remove instruction files older than max_age_hours."""
    instructions_path = Path(INSTRUCTIONS_DIR)
    if not instructions_path.exists():
        return 0

    now = datetime.now()
    cleaned = 0
    try:
        for f in instructions_path.glob("vibe_instructions_*.md"):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if (now - mtime).total_seconds() > max_age_hours * 3600:
                    f.unlink()
                    cleaned += 1
            except Exception as e:
                logger.debug(f"Failed to cleanup {f.name}: {e}")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

    if cleaned > 0:
        logger.info(f"Cleaned {cleaned} old instruction files")
    return cleaned


def handle_long_prompt(prompt: str, cwd: str | None = None) -> tuple[str, str | None]:
    """
    Handle long prompts by offloading to a file.
    Returns (final_prompt_arg, file_path_to_cleanup)
    """
    if len(prompt) <= 2000:
        return prompt, None

    try:
        os.makedirs(INSTRUCTIONS_DIR, exist_ok=True)

        timestamp = int(datetime.now().timestamp())
        unique_id = uuid.uuid4().hex[:6]
        filename = f"vibe_instructions_{timestamp}_{unique_id}.md"
        filepath = os.path.join(INSTRUCTIONS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# VIBE INSTRUCTIONS\n\n")
            f.write(prompt)

        logger.debug(f"Large prompt ({len(prompt)} chars) stored at {filepath}")

        # Return a reference to the file
        return f"Read and execute the instructions from file: {filepath}", filepath

    except Exception as e:
        logger.warning(f"Failed to offload prompt: {e}")
        # Fallback: truncate if necessary
        if len(prompt) > 10000:
            return prompt[:10000] + "\n[TRUNCATED]", None
        return prompt, None


async def run_vibe_subprocess(
    argv: list[str],
    cwd: str | None,
    timeout_s: float,
    env: dict[str, str] | None = None,
    ctx: Context | None = None,
    prompt_preview: str | None = None,
) -> dict[str, Any]:
    """
    Execute Vibe CLI subprocess with streaming output.

    Returns:
        Dict with keys: success, stdout, stderr, returncode, command
    """
    config = get_vibe_config()

    # Prepare environment
    process_env = os.environ.copy()
    process_env.update(config.get_environment())
    if env:
        process_env.update({k: str(v) for k, v in env.items()})

    logger.debug(f"[VIBE] Executing: {' '.join(argv)}")

    async def emit_log(level: Literal["debug", "error", "info", "warning"], message: str) -> None:
        if not ctx:
            return
        try:
            await ctx.log(level, message, logger_name="vibe_mcp")
        except Exception as e:
            logger.debug(f"[VIBE] Failed to send log to client: {e}")

    # Повідомлення про початок
    if prompt_preview:
        await emit_log("info", f"🚀 [VIBE-LIVE] Запуск Vibe: {prompt_preview[:80]}...")

    try:
        full_argv = argv

        process = await asyncio.create_subprocess_exec(
            *full_argv,
            cwd=cwd or VIBE_WORKSPACE,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        async def read_stream_with_logging(
            stream: asyncio.StreamReader, chunks: list[bytes], stream_name: str
        ) -> None:
            """Read from stream, log important lines, collect output."""
            buffer = b""

            async def handle_line(line: str) -> None:
                if not line:
                    return

                # Filter out terminal control characters and TUI artifacts
                if any(c < "\x20" for c in line if c not in "\t\n\r"):
                    line = "".join(c for c in line if c >= "\x20" or c in "\t\n\r")

                if not line:
                    return

                # Try to parse as JSON for structured logging
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("role") and obj.get("content"):
                        preview = str(obj["content"])[:200]
                        # Форматування для UI
                        if obj["role"] == "assistant":
                            message = f"🧠 [VIBE-THOUGHT] {preview}"
                        elif obj["role"] == "tool":
                            message = f"🔧 [VIBE-ACTION] {preview}"
                        else:
                            message = f"💬 [VIBE-GEN] {preview}"

                        logger.info(message)
                        await emit_log("info", message)
                        return
                except json.JSONDecodeError:
                    pass

                # Regular log line - filter out TUI spam
                spam_triggers = [
                    "Welcome to",
                    "│",
                    "╭",
                    "╮",
                    "╰",
                    "─",
                    "──",
                    "[2K",
                    "[1A",
                    "Press Enter",
                    "↵",
                ]
                if any(t in line for t in spam_triggers):
                    return

                # Живий стрім для UI
                if len(line) < 1000:
                    # Форматуємо для ExecutionLog компонента
                    if "Thinking" in line or "Planning" in line:
                        formatted = f"🧠 [VIBE-THOUGHT] {line}"
                    elif "Running" in line or "Executing" in line:
                        formatted = f"🔧 [VIBE-ACTION] {line}"
                    else:
                        formatted = f"⚡ [VIBE-LIVE] {line}"

                    logger.debug(f"[VIBE_{stream_name}] {line}")
                    level = "warning" if stream_name == "ERR" else "info"
                    await emit_log(level, formatted)

            try:
                while True:
                    data = await stream.read(8192)
                    if not data:
                        break

                    chunks.append(data)
                    buffer += data

                    # Process complete lines
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = strip_ansi(line_bytes.decode(errors="replace")).strip()
                        await handle_line(line)

                if buffer:
                    line = strip_ansi(buffer.decode(errors="replace")).strip()
                    await handle_line(line)

            except TimeoutError:
                logger.warning(f"[VIBE] Read timeout on {stream_name} after {timeout_s}s")
            except Exception as e:
                logger.error(f"[VIBE] Stream reading error ({stream_name}): {e}")

        # Read both streams concurrently
        try:
            from collections.abc import Awaitable

            tasks: list[Awaitable[Any]] = []
            if process.stdout:
                tasks.append(read_stream_with_logging(process.stdout, stdout_chunks, "OUT"))
            if process.stderr:
                tasks.append(read_stream_with_logging(process.stderr, stderr_chunks, "ERR"))
            tasks.append(process.wait())

            await asyncio.wait_for(
                asyncio.gather(*tasks),  # pyrefly: ignore[bad-argument-type]
                timeout=timeout_s + 20,  # Add buffer for graceful shutdown
            )
            # Повідомлення про успіх
            await emit_log("info", "✅ [VIBE-LIVE] Vibe завершив роботу успішно")
        except TimeoutError:
            logger.warning(f"[VIBE] Process timeout ({timeout_s}s), terminating")
            await emit_log("warning", f"⏱️ [VIBE-LIVE] Перевищено timeout ({timeout_s}s)")
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()

            stdout_str = strip_ansi(b"".join(stdout_chunks).decode(errors="replace"))
            stderr_str = strip_ansi(b"".join(stderr_chunks).decode(errors="replace"))

            # Final cleanup of the strings
            stdout_str = "".join(c for c in stdout_str if c >= "\x20" or c in "\t\n\r")
            stderr_str = "".join(c for c in stderr_str if c >= "\x20" or c in "\t\n\r")

            return {
                "success": False,
                "error": f"Vibe execution timed out after {timeout_s}s",
                "returncode": -1,
                "stdout": truncate_output(stdout_str),
                "stderr": truncate_output(stderr_str),
                "command": argv,
            }

        stdout = strip_ansi(b"".join(stdout_chunks).decode(errors="replace"))
        stderr = strip_ansi(b"".join(stderr_chunks).decode(errors="replace"))

        # Final cleanup of the strings
        stdout = "".join(c for c in stdout if c >= "\x20" or c in "\t\n\r")
        stderr = "".join(c for c in stderr if c >= "\x20" or c in "\t\n\r")

        logger.info(f"[VIBE] Process completed with exit code: {process.returncode}")

        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": truncate_output(stdout),
            "stderr": truncate_output(stderr),
            "command": argv,
        }

    except FileNotFoundError as e:
        error_msg = f"Vibe binary not found: {argv[0]}"
        logger.error(f"[VIBE] {error_msg}: {e}")
        return {
            "success": False,
            "error": error_msg,
            "command": argv,
        }

    except Exception as e:
        error_msg = f"Subprocess error: {e}"
        logger.error(f"[VIBE] {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "command": argv,
        }


# =============================================================================
# MCP TOOLS - CORE (6 tools)
# =============================================================================


@server.tool()
async def vibe_which(ctx: Context) -> dict[str, Any]:
    """
    Locate the Vibe CLI binary and report its version and configuration.

    Returns:
        Dict with 'binary' path, 'version', current 'model', and 'mode'
    """
    vibe_path = resolve_vibe_binary()
    if not vibe_path:
        logger.warning("[VIBE] Binary not found on PATH")
        return {
            "success": False,
            "error": f"Vibe CLI not found (binary='{VIBE_BINARY}')",
        }

    logger.debug(f"[VIBE] Found binary at: {vibe_path}")

    try:
        process = await asyncio.create_subprocess_exec(
            vibe_path,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        version = stdout.decode().strip() if process.returncode == 0 else "unknown"
    except Exception as e:
        logger.warning(f"Failed to get Vibe version: {e}")
        version = "unknown"

    config = get_vibe_config()

    return {
        "success": True,
        "binary": vibe_path,
        "version": version,
        "active_model": _current_model or config.active_model,
        "mode": _current_mode.value,
        "available_models": [m.alias for m in config.get_available_models()],
    }


@server.tool()
async def vibe_prompt(
    ctx: Context,
    prompt: str,
    cwd: str | None = None,
    timeout_s: float | None = None,
    # Enhanced options
    model: str | None = None,
    agent: str | None = None,
    mode: str | None = None,
    session_id: str | None = None,
    enabled_tools: list[str] | None = None,
    disabled_tools: list[str] | None = None,
    max_turns: int | None = None,
    max_price: float | None = None,
    output_format: str = "streaming",
) -> dict[str, Any]:
    """
    Send a prompt to Vibe AI agent in programmatic mode.

    The PRIMARY tool for interacting with Vibe. Executes in CLI mode with
    structured output. All execution is logged and visible.

    Args:
        prompt: The message/query for Vibe AI (Mistral-powered)
        cwd: Working directory for execution (default: vibe_workspace)
        timeout_s: Timeout in seconds (default from config)
        model: Model alias to use (overrides active_model)
        agent: Agent profile name (loads from agents directory)
        mode: Operational mode (plan/auto-approve/accept-edits)
        session_id: Session ID to resume
        enabled_tools: Additional tools to enable (glob/regex patterns)
        disabled_tools: Additional tools to disable (glob/regex patterns)
        max_turns: Maximum conversation turns
        max_price: Maximum cost limit in dollars
        output_format: Output format (streaming/json/text)

    Returns:
        Dict with 'success', 'stdout', 'stderr', 'returncode', 'parsed_response'
    """
    prepare_workspace_and_instructions()

    vibe_path = resolve_vibe_binary()
    if not vibe_path:
        return {
            "success": False,
            "error": "Vibe CLI not found on PATH",
        }

    config = get_vibe_config()
    eff_timeout = timeout_s if timeout_s is not None else config.timeout_s
    eff_cwd = cwd or VIBE_WORKSPACE

    # Ensure workspace exists
    os.makedirs(eff_cwd, exist_ok=True)

    # Check network before proceeding if it's an AI prompt
    if not await is_network_available():
        return {
            "success": False,
            "error": "Mistral API is unreachable. Please check your internet connection.",
            "returncode": -2,
        }

    final_prompt, prompt_file_to_clean = handle_long_prompt(prompt, eff_cwd)

    try:
        # Determine effective mode
        effective_mode = AgentMode(mode) if mode else _current_mode

        # Build command using config
        argv = [
            vibe_path,
            *config.to_cli_args(
                prompt=final_prompt,
                mode=effective_mode,
                model=model or _current_model,
                agent=agent,
                session_id=session_id,
                max_turns=max_turns,
                max_price=max_price,
                output_format=output_format,
            ),
        ]

        logger.info(f"[VIBE] Executing prompt: {prompt[:50]}... (timeout={eff_timeout}s)")

        result = await run_vibe_subprocess(
            argv=argv,
            cwd=eff_cwd,
            timeout_s=eff_timeout,
            ctx=ctx,
            prompt_preview=prompt,
        )

        # Try to parse JSON response
        if result.get("success") and result.get("stdout"):
            try:
                result["parsed_response"] = json.loads(result["stdout"])
            except json.JSONDecodeError:
                # Try to extract JSON from streaming format
                lines = result["stdout"].split("\n")
                json_lines = [line for line in lines if line.strip().startswith("{")]
                if json_lines:
                    try:
                        result["parsed_response"] = json.loads(json_lines[-1])
                    except json.JSONDecodeError:
                        result["parsed_response"] = None

        return result

    finally:
        # Cleanup temporary file
        if prompt_file_to_clean and os.path.exists(prompt_file_to_clean):
            try:
                os.remove(prompt_file_to_clean)
                logger.debug(f"Cleaned up prompt file: {prompt_file_to_clean}")
            except Exception as e:
                logger.warning(f"Failed to cleanup prompt file: {e}")


@server.tool()
async def vibe_analyze_error(
    ctx: Context,
    error_message: str,
    file_path: str | None = None,
    log_context: str | None = None,
    recovery_history: list[dict[str, Any]] | str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
    auto_fix: bool = True,
) -> dict[str, Any]:
    """
    Deep error analysis and optional auto-fix using Vibe AI.

    Designed for self-healing scenarios when the system encounters errors
    it cannot resolve. Vibe acts as a Senior Engineer.

    Args:
        error_message: The error message or stack trace
        file_path: Path to the file with the error (if known)
        log_context: Recent log entries for context
        recovery_history: List of past recovery attempts or a summary string
        cwd: Working directory
        timeout_s: Timeout in seconds (default: 600)
        auto_fix: Automatically apply fixes (default: True)

    Returns:
        Analysis with root cause, suggested or applied fixes, and verification
    """
    prepare_workspace_and_instructions()

    prompt_parts = [
        "SYSTEM: You are the Senior Self-Healing Engineer for AtlasTrinity.",
        "",
        "DATABASE SCHEMA:",
        "- sessions: id, started_at, ended_at",
        "- tasks: id, session_id, goal, status, created_at",
        "- task_steps: id, task_id, sequence_number, action, tool, status, error_message",
        "- tool_executions: id, step_id, server_name, tool_name, arguments, result",
        "",
        f"CONTEXT: System Root: {SYSTEM_ROOT} | Project: {cwd or VIBE_WORKSPACE}",
        "",
        f"ERROR MESSAGE:\n{error_message}",
    ]

    if log_context:
        prompt_parts.append(f"\nRECENT LOGS:\n{log_context}")

    if recovery_history:
        if isinstance(recovery_history, list):
            history_str = "\n".join(
                [
                    f"- Attempt {i + 1}: {a.get('action', 'Unknown')} | Result: {a.get('status', 'Unknown')} | Error: {a.get('error_message', 'N/A')}"
                    for i, a in enumerate(recovery_history)
                ]
            )
            prompt_parts.append(
                f"\nPAST ATTEMPTS:\n{history_str}\n(Avoid repeating failed strategies)"
            )
        else:
            prompt_parts.append(
                f"\nPAST ATTEMPTS:\n{recovery_history}\n(Avoid repeating failed strategies)"
            )

    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()[:5000]  # Limit
                prompt_parts.append(f"\nFILE: {file_path}\n```\n{content}\n```")
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")

    if auto_fix:
        prompt_parts.extend(
            [
                "",
                "INSTRUCTIONS:",
                "1. Perform Root Cause Analysis (RCA)",
                "2. Create a fix strategy",
                "3. Execute the fix (edit code, run commands)",
                "4. Verify the fix works",
                "5. Report the results",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "",
                "Analyze and suggest fixes without applying them.",
            ]
        )

    prompt = "\n".join(prompt_parts)

    logger.info(f"[VIBE] Analyzing error (auto_fix={auto_fix})")

    return await vibe_prompt(
        ctx=ctx,
        prompt=prompt,
        cwd=cwd,
        timeout_s=timeout_s or DEFAULT_TIMEOUT_S,
        model=AGENT_MODEL_OVERRIDE,
        mode="auto-approve" if auto_fix else "plan",
        max_turns=15,
    )


@server.tool()
async def vibe_implement_feature(
    ctx: Context,
    goal: str,
    context_files: list[str] | None = None,
    constraints: str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = 1200,
) -> dict[str, Any]:
    """
    Deep coding mode: Implements a complex feature or refactoring.

    Vibe acts as a Senior Architect to plan, implement, and verify changes.

    Args:
        goal: High-level objective (e.g., "Add user profile page with API and DB")
        context_files: List of relevant file paths
        constraints: Technical constraints or guidelines
        cwd: Working directory
        timeout_s: Timeout (default: 1200s for deep work)

    Returns:
        Implementation report with changed files and verification
    """
    prepare_workspace_and_instructions()

    # Gather file contents
    file_contents = []
    if context_files:
        for fpath in context_files:
            if os.path.exists(fpath):
                try:
                    with open(fpath, encoding="utf-8") as f:
                        content = f.read()[:5000]  # Limit per file
                        file_contents.append(f"FILE: {fpath}\n```\n{content}\n```")
                except Exception as e:
                    file_contents.append(f"FILE: {fpath} (Error: {e})")
            else:
                file_contents.append(f"FILE: {fpath} (Not found, will create)")

    context_str = "\n\n".join(file_contents) if file_contents else "(No files provided)"

    prompt = f"""
SYSTEM: You are the Senior Software Architect and Lead Developer for AtlasTrinity.
ROLE: Implement a complex feature efficiently and robustly.

GOAL: {goal}

CONTEXT FILES:
{context_str}

CONSTRAINTS:
{constraints or "Standard project guidelines apply."}

System Root: {SYSTEM_ROOT}
Project Directory: {cwd or VIBE_WORKSPACE}

INSTRUCTIONS:
1. PLAN: Analyze the goal and files
2. IMPLEMENT: Edit necessary files, handle imports and dependencies
3. VERIFY: Run checks to ensure no syntax errors
4. REPORT: List exactly which files were modified and confirm success

EXECUTE NOW.
"""

    return await vibe_prompt(
        ctx=ctx,
        prompt=prompt,
        cwd=cwd,
        timeout_s=timeout_s or 1200,
        model=AGENT_MODEL_OVERRIDE,
        mode="auto-approve",
        max_turns=30,
    )


@server.tool()
async def vibe_code_review(
    ctx: Context,
    file_path: str,
    focus_areas: str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """
    Request a code review from Vibe AI for a specific file.

    Args:
        file_path: Path to the file to review
        focus_areas: Specific areas to focus on (e.g., "security", "performance")
        cwd: Working directory
        timeout_s: Timeout in seconds (default: 300)

    Returns:
        Code review analysis with suggestions
    """
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()[:10000]  # Limit
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not read file: {e}",
        }

    prompt_parts = [
        f"CODE REVIEW REQUEST: {file_path}",
        "",
        f"FILE CONTENT:\n```\n{content}\n```",
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

    return await vibe_prompt(
        ctx=ctx,
        prompt="\n".join(prompt_parts),
        cwd=cwd,
        timeout_s=timeout_s or 300,
        model=AGENT_MODEL_OVERRIDE,
        mode="plan",  # Read-only mode
        max_turns=5,
    )


@server.tool()
async def vibe_smart_plan(
    ctx: Context,
    objective: str,
    context: str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """
    Generate a smart execution plan for a complex objective.

    Args:
        objective: The goal or task to plan for
        context: Additional context (existing code, constraints, etc.)
        cwd: Working directory
        timeout_s: Timeout in seconds (default: 300)

    Returns:
        Structured plan with steps, actions, tools, and verification criteria
    """
    prompt_parts = [
        "CREATE A DETAILED EXECUTION PLAN",
        "",
        f"OBJECTIVE: {objective}",
    ]

    if context:
        prompt_parts.append(f"\nCONTEXT:\n{context}")

    prompt_parts.extend(
        [
            "",
            "For each step, specify:",
            "- Action to perform",
            "- Required tools/commands",
            "- Expected outcome",
            "- Verification criteria",
        ]
    )

    return await vibe_prompt(
        ctx=ctx,
        prompt="\n".join(prompt_parts),
        cwd=cwd,
        timeout_s=timeout_s or 300,
        mode="plan",
        max_turns=5,
    )


# =============================================================================
# MCP TOOLS - CONFIGURATION (5 new tools)
# =============================================================================


@server.tool()
async def vibe_get_config(ctx: Context) -> dict[str, Any]:
    """
    Get the current Vibe configuration state.

    Returns:
        Current configuration including active model, mode, providers, and models
    """
    config = get_vibe_config()

    return {
        "success": True,
        "active_model": _current_model or config.active_model,
        "mode": _current_mode.value,
        "default_mode": config.default_mode.value,
        "max_turns": config.max_turns,
        "max_price": config.max_price,
        "timeout_s": config.timeout_s,
        "providers": [
            {
                "name": p.name,
                "api_base": p.api_base,
                "available": p.is_available(),
            }
            for p in config.providers
        ],
        "models": [
            {
                "alias": m.alias,
                "name": m.name,
                "provider": m.provider,
                "temperature": m.temperature,
            }
            for m in config.models
        ],
        "available_models": [m.alias for m in config.get_available_models()],
        "enabled_tools": config.enabled_tools,
        "disabled_tools": config.disabled_tools,
    }


@server.tool()
async def vibe_configure_model(
    ctx: Context,
    model_alias: str,
    persist: bool = False,
) -> dict[str, Any]:
    """
    Switch the active model for Vibe operations.

    Args:
        model_alias: Alias of the model to use (from models list)
        persist: If True, update the config file (not yet implemented)

    Returns:
        Confirmation with the new active model
    """
    global _current_model

    config = get_vibe_config()
    model = config.get_model_by_alias(model_alias)

    if not model:
        available = [m.alias for m in config.models]
        return {
            "success": False,
            "error": f"Model '{model_alias}' not found",
            "available_models": available,
        }

    # Check if provider is available
    provider = config.get_provider(model.provider)
    if not provider or not provider.is_available():
        return {
            "success": False,
            "error": f"Provider '{model.provider}' is not available (missing API key)",
            "hint": f"Set {provider.api_key_env_var if provider else 'API_KEY'} environment variable",
        }

    _current_model = model_alias
    logger.info(f"[VIBE] Switched active model to: {model_alias}")

    return {
        "success": True,
        "active_model": model_alias,
        "model_name": model.name,
        "provider": model.provider,
        "temperature": model.temperature,
    }


@server.tool()
async def vibe_set_mode(
    ctx: Context,
    mode: str,
) -> dict[str, Any]:
    """
    Change the operational mode for Vibe.

    Args:
        mode: Operational mode - "default", "plan", "accept-edits", or "auto-approve"
            - default: Requires confirmation for tool executions
            - plan: Read-only mode for exploration
            - accept-edits: Auto-approves file edit tools only
            - auto-approve: Auto-approves all tool executions

    Returns:
        Confirmation with the new mode
    """
    global _current_mode

    try:
        new_mode = AgentMode(mode)
    except ValueError:
        return {
            "success": False,
            "error": f"Invalid mode: '{mode}'",
            "valid_modes": [m.value for m in AgentMode],
        }

    _current_mode = new_mode
    logger.info(f"[VIBE] Changed operational mode to: {mode}")

    return {
        "success": True,
        "mode": mode,
        "description": {
            "default": "Requires confirmation for tool executions",
            "plan": "Read-only mode for exploration",
            "accept-edits": "Auto-approves file edit tools only",
            "auto-approve": "Auto-approves all tool executions",
        }.get(mode, "Unknown"),
    }


@server.tool()
async def vibe_configure_provider(
    ctx: Context,
    name: str,
    api_base: str,
    api_key_env_var: str,
    api_style: str = "openai",
    backend: str = "generic",
) -> dict[str, Any]:
    """
    Add or update a provider configuration (runtime only).

    Args:
        name: Provider identifier
        api_base: Base URL for API calls
        api_key_env_var: Environment variable for API key
        api_style: API style - "mistral", "openai", or "anthropic"
        backend: Backend implementation - "mistral", "generic", or "anthropic"

    Returns:
        Confirmation with provider details
    """
    config = get_vibe_config()

    try:
        new_provider = ProviderConfig(
            name=name,
            api_base=api_base,
            api_key_env_var=api_key_env_var,
            api_style=api_style,  # type: ignore
            backend=backend,  # type: ignore
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid provider configuration: {e}",
        }

    # Check if provider already exists
    existing = config.get_provider(name)
    if existing:
        # Update existing (remove and re-add)
        config.providers = [p for p in config.providers if p.name != name]

    config.providers.append(new_provider)
    logger.info(f"[VIBE] Added/updated provider: {name}")

    return {
        "success": True,
        "provider": name,
        "api_base": api_base,
        "available": new_provider.is_available(),
        "note": "This change is runtime-only. Add to vibe_config.toml for persistence.",
    }


@server.tool()
async def vibe_session_resume(
    ctx: Context,
    session_id: str,
    prompt: str | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """
    Resume a previous Vibe session.

    Args:
        session_id: Session ID to resume (partial match supported)
        prompt: Optional new prompt to continue with
        cwd: Working directory
        timeout_s: Timeout in seconds

    Returns:
        Result of the resumed session
    """
    # Verify session exists
    target_path = None

    # Search in session directory
    if VIBE_SESSION_DIR.exists():
        files = list(VIBE_SESSION_DIR.glob(f"*{session_id}*.json"))
        if files:
            target_path = files[0]

    if not target_path:
        return {
            "success": False,
            "error": f"Session '{session_id}' not found",
            "hint": "Use vibe_list_sessions to see available sessions",
        }

    # Extract full session ID from filename
    full_session_id = target_path.stem.replace("session_", "")

    # Use vibe_prompt with session continuation
    return await vibe_prompt(
        ctx=ctx,
        prompt=prompt or "Continue from where we left off.",
        cwd=cwd,
        timeout_s=timeout_s,
        session_id=full_session_id,
    )


# =============================================================================
# MCP TOOLS - UTILITY (5 tools)
# =============================================================================


@server.tool()
async def vibe_ask(
    ctx: Context,
    question: str,
    cwd: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """
    Ask Vibe AI a quick question (read-only, no tool execution).

    Args:
        question: The question to ask
        cwd: Working directory
        timeout_s: Timeout in seconds (default: 300)

    Returns:
        AI response without file modifications
    """
    return await vibe_prompt(
        ctx=ctx,
        prompt=question,
        cwd=cwd,
        timeout_s=timeout_s or 300,
        mode="plan",
        max_turns=3,
        output_format="json",
    )


@server.tool()
async def vibe_execute_subcommand(
    ctx: Context,
    subcommand: str,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout_s: float | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Execute a specific Vibe CLI subcommand (utility operations).

    For AI interactions, use vibe_prompt() instead.

    Allowed subcommands:
        list-editors, list-modules, run, enable, disable, install,
        agent-reset, agent-on, agent-off, vibe-status, vibe-continue,
        vibe-cancel, vibe-help, eternal-engine, screenshots

    Args:
        subcommand: The Vibe subcommand
        args: Optional arguments
        cwd: Working directory
        timeout_s: Timeout in seconds
        env: Additional environment variables

    Returns:
        Command output and exit code
    """
    vibe_path = resolve_vibe_binary()
    if not vibe_path:
        return {"success": False, "error": "Vibe CLI not found"}

    sub = (subcommand or "").strip()
    if not sub:
        return {"success": False, "error": "Missing subcommand"}

    if sub in BLOCKED_SUBCOMMANDS:
        return {
            "success": False,
            "error": f"Subcommand '{sub}' is interactive and blocked",
            "suggestion": "Use vibe_prompt() for AI interactions",
        }

    if sub not in ALLOWED_SUBCOMMANDS:
        return {
            "success": False,
            "error": f"Unknown subcommand: '{sub}'",
            "allowed": sorted(ALLOWED_SUBCOMMANDS),
        }

    argv = [vibe_path, sub]
    if args:
        # Filter out interactive arguments
        clean_args = [str(a) for a in args if a != "--no-tui"]
        argv.extend(clean_args)

    # Create preview from subcommand and args
    preview = f"{sub} {' '.join(str(a) for a in (args or []))[:50]}"

    return await run_vibe_subprocess(
        argv=argv,
        cwd=cwd,
        timeout_s=timeout_s or DEFAULT_TIMEOUT_S,
        env=env,
        ctx=ctx,
        prompt_preview=preview,
    )


@server.tool()
async def vibe_list_sessions(ctx: Context, limit: int = 10) -> dict[str, Any]:
    """
    List recent Vibe session logs with metrics.

    Useful for tracking costs, context size, and session IDs for resuming.

    Args:
        limit: Number of sessions to return (default: 10)

    Returns:
        List of recent sessions with metadata
    """
    if not VIBE_SESSION_DIR.exists():
        return {
            "success": False,
            "error": f"Session directory not found at {VIBE_SESSION_DIR}",
        }

    try:
        files = sorted(
            VIBE_SESSION_DIR.glob("session_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )[:limit]

        sessions = []
        for f in files:
            try:
                with open(f, encoding="utf-8") as jf:
                    data = json.load(jf)
                    meta = data.get("metadata", {})
                    stats = meta.get("stats", {})

                    sessions.append(
                        {
                            "session_id": meta.get("session_id"),
                            "timestamp": meta.get("start_time"),
                            "steps": stats.get("steps", 0),
                            "prompt_tokens": stats.get("session_prompt_tokens", 0),
                            "completion_tokens": stats.get("session_completion_tokens", 0),
                            "file": f.name,
                        }
                    )
            except Exception as e:
                logger.debug(f"Failed to parse session {f.name}: {e}")

        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
        }

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {
            "success": False,
            "error": f"Failed to list sessions: {e}",
        }


@server.tool()
async def vibe_session_details(ctx: Context, session_id_or_file: str) -> dict[str, Any]:
    """
    Get full details of a specific Vibe session.

    Args:
        session_id_or_file: Session ID or filename

    Returns:
        Full session details including history and token counts
    """
    target_path = None

    # Check absolute path
    if os.path.isabs(session_id_or_file) and os.path.exists(session_id_or_file):
        target_path = Path(session_id_or_file)

    # Check in session directory
    elif (VIBE_SESSION_DIR / session_id_or_file).exists():
        target_path = VIBE_SESSION_DIR / session_id_or_file

    # Search by pattern
    else:
        files = list(VIBE_SESSION_DIR.glob(f"*{session_id_or_file}*.json"))
        if files:
            target_path = files[0]

    if not target_path:
        return {
            "success": False,
            "error": f"Session '{session_id_or_file}' not found",
        }

    try:
        with open(target_path, encoding="utf-8") as f:
            data = json.load(f)
            return {
                "success": True,
                "data": data,
            }
    except Exception as e:
        logger.error(f"Failed to read session: {e}")
        return {
            "success": False,
            "error": f"Failed to read session: {e}",
        }


@server.tool()
async def vibe_reload_config(ctx: Context) -> dict[str, Any]:
    """
    Reload the Vibe configuration from disk.

    Returns:
        New configuration summary
    """
    global _current_mode, _current_model

    try:
        config = reload_vibe_config()

        # Reset runtime overrides
        _current_mode = config.default_mode
        _current_model = None

        return {
            "success": True,
            "active_model": config.active_model,
            "mode": config.default_mode.value,
            "providers_count": len(config.providers),
            "models_count": len(config.models),
        }
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        return {
            "success": False,
            "error": f"Failed to reload config: {e}",
        }


# =============================================================================
# MCP TOOLS - DATABASE (2 tools)
# =============================================================================


@server.tool()
async def vibe_check_db(ctx: Context, query: str) -> dict[str, Any]:
    """
    Execute a read-only SQL query against the AtlasTrinity database.

    Use this to inspect task execution history, tool results, and system state.

    SCHEMA:
    - sessions: id, started_at, ended_at
    - tasks: id, session_id, goal, status, created_at
    - task_steps: id, task_id, sequence_number, action, tool, status, error_message
    - tool_executions: id, step_id, server_name, tool_name, arguments, result
    - logs: timestamp, level, source, message

    Args:
        query: SQL SELECT query (SELECT queries only for safety)

    Returns:
        Query results as list of dictionaries
    """
    from sqlalchemy import text

    from src.brain.db.manager import db_manager

    # Prevent destructive operations
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"]
    if any(f in query.upper() for f in forbidden):
        return {
            "success": False,
            "error": "Only SELECT queries are allowed for safety",
        }

    # Use central DB manager when available
    try:
        await db_manager.initialize()
        if not db_manager.available:
            return {"success": False, "error": "Database not initialized"}

        session = await db_manager.get_session()
        try:
            res = await session.execute(text(query))
            rows = [dict(r) for r in res.mappings().all()]
            return {"success": True, "count": len(rows), "data": rows}
        finally:
            await session.close()

    except Exception as e:
        logger.error(f"Database query error: {e}")
        return {"success": False, "error": str(e)}


@server.tool()
async def vibe_get_system_context(ctx: Context) -> dict[str, Any]:
    """
    Retrieve current operational context from the database.

    Helps Vibe focus on the current state before performing deep analysis.

    Returns:
        Current session, recent tasks, and errors
    """
    from sqlalchemy import text

    from src.brain.db.manager import db_manager

    try:
        await db_manager.initialize()
        if not db_manager.available:
            return {"success": False, "error": "Database not initialized"}

        db_session = await db_manager.get_session()
        try:
            # Latest session
            res = await db_session.execute(
                text("SELECT id, started_at FROM sessions ORDER BY started_at DESC LIMIT 1")
            )
            session_row = res.mappings().first()
            session_id = str(session_row["id"]) if session_row else None

            # Latest tasks
            tasks = []
            if session_id:
                tasks_res = await db_session.execute(
                    text(
                        "SELECT id, goal, status, created_at FROM tasks WHERE session_id = :sid ORDER BY created_at DESC LIMIT 5"
                    ),
                    {"sid": session_id},
                )
                tasks = [dict(r) for r in tasks_res.mappings().all()]

            # Recent errors
            errors_res = await db_session.execute(
                text(
                    "SELECT timestamp, source, message FROM logs WHERE level IN ('ERROR', 'WARNING') ORDER BY timestamp DESC LIMIT 5"
                )
            )
            errors = [dict(r) for r in errors_res.mappings().all()]

            return {
                "success": True,
                "current_session_id": session_id,
                "recent_tasks": tasks,
                "recent_errors": errors,
                "system_root": SYSTEM_ROOT,
                "project_root": VIBE_WORKSPACE,
            }
        finally:
            await db_session.close()
    except Exception as e:
        logger.error(f"Database query error in vibe_get_system_context: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logger.info("[VIBE] MCP Server starting (v3.0 Hyper-Refactored)...")
    prepare_workspace_and_instructions()
    cleanup_old_instructions()

    # Pre-load configuration
    try:
        config = get_vibe_config()
        logger.info(
            f"[VIBE] Configuration loaded: {len(config.models)} models, {len(config.providers)} providers"
        )
    except Exception as e:
        logger.warning(f"[VIBE] Could not load configuration: {e}")

    try:
        server.run()
    except (BrokenPipeError, KeyboardInterrupt):
        logger.info("[VIBE] Server shutdown requested")
        sys.exit(0)
    except BaseException as e:
        # Handle ExceptionGroups that may contain BrokenPipeError
        def is_broken_pipe(exc: BaseException) -> bool:
            if isinstance(exc, BrokenPipeError) or "Broken pipe" in str(exc):
                return True
            if hasattr(exc, "exceptions"):
                return any(is_broken_pipe(e) for e in exc.exceptions)
            return False

        if is_broken_pipe(e):
            sys.exit(0)

        logger.error(f"[VIBE] Unexpected error: {e}")
        raise
