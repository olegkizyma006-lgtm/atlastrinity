import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.server import FastMCP

# Initialize FastMCP server
server = FastMCP("devtools-server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@server.tool()
def devtools_check_mcp_health() -> dict[str, Any]:
    """
    Run the system-wide MCP health check script.
    Ping all enabled servers and report their status, response time, and tool counts.
    """
    script_path = PROJECT_ROOT / "scripts" / "check_mcp_health.py"

    if not script_path.exists():
        return {"error": f"Health check script not found at {script_path}"}

    try:
        # Run scripts/check_mcp_health.py --json
        cmd = [sys.executable, str(script_path), "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        output = result.stdout.strip()
        if not output:
            return {"error": "Health check returned empty output", "stderr": result.stderr}

        try:
            data = json.loads(output)
            return data
        except json.JSONDecodeError:
            return {"error": "Failed to parse health check JSON", "raw_output": output}

    except Exception as e:
        return {"error": str(e)}


@server.tool()
def devtools_launch_inspector(server_name: str) -> dict[str, Any]:
    """
    Launch the official MCP Inspector for a specific server (Tier 1-4).
    This starts a background process and returns a URL (localhost) to open in the browser.

    Args:
        server_name: The name of the server to inspect (e.g., 'memory', 'vibe', 'filesystem').

    Note: The inspector process continues running in the background.
    """
    # Load active MCP config to find command
    config_path = Path.home() / ".config" / "atlastrinity" / "mcp" / "config.json"
    if not config_path.exists():
        # Fallback to template if active not found (unlikely in prod but helpful for dev)
        config_path = PROJECT_ROOT / "src" / "mcp_server" / "config.json.template"
        
    if not config_path.exists():
        return {"error": "MCP Configuration not found"}

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        server_config = config.get("mcpServers", {}).get(server_name)
        if not server_config:
            return {"error": f"Server '{server_name}' not found in configuration."}

        command = server_config.get("command")
        args = server_config.get("args", [])
        env_vars = server_config.get("env", {})

        # Construct inspector command
        # npx @modelcontextprotocol/inspector <command> <args>
        inspector_cmd = ["npx", "@modelcontextprotocol/inspector", command, *args]

        # Prepare environment
        env = os.environ.copy()
        # Resolve variables in args/env (basic resolution)
        # NOTE: This is a simplified resolution. For full resolution, we'd need mcp_manager logic.
        # But commonly used vars are usually just HOME or PROJECT_ROOT.

        # Basic substitution for '${HOME}' and '${PROJECT_ROOT}' in args
        resolved_inspector_cmd = []
        for arg in inspector_cmd:
            arg = arg.replace("${HOME}", str(Path.home()))
            arg = arg.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
            resolved_inspector_cmd.append(arg)

        # Add server-specific env vars
        for k, v in env_vars.items():
            val = v.replace("${GITHUB_TOKEN}", env.get("GITHUB_TOKEN", ""))
            env[k] = val

        # Start detached process
        # We redirect stdout/stderr to capture the URL, but we need to be careful not to block.
        # Ideally, we start it, wait a second to scrape the URL from stderr, then let it run.

        proc = subprocess.Popen(
            resolved_inspector_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,  # Detach
        )

        # Peek at output to find URL (inspector prints to stderr usually)
        # We'll wait up to 5 seconds
        import time

        for _ in range(10):
            if proc.poll() is not None:
                # Process died
                out, err = proc.communicate()
                return {
                    "error": "Inspector process exited immediately",
                    "stdout": out,
                    "stderr": err,
                }

            # We can't easily read without blocking unless we use threads or fancy non-blocking I/O.
            # Simple approach: Return success and tell user to check output or assume standard port.
            # But the user wants the URL.
            # Let's try to assume it works and return a generic message,
            # OR better: The inspector usually prints "Inspector is running at http://localhost:xxxx"

            time.sleep(0.5)

        # If it's still running, we assume success.
        return {
            "success": True,
            "message": f"Inspector launched for '{server_name}'.",
            "pid": proc.pid,
            "note": "Please check http://localhost:5173 (default) or check terminal output if visible.",
        }

    except Exception as e:
        return {"error": str(e)}


@server.tool()
def devtools_validate_config() -> dict[str, Any]:
    """
    Validate the syntax and basic structure of the local MCP configuration file.
    """
    config_path = Path.home() / ".config" / "atlastrinity" / "mcp" / "config.json"

    if not config_path.exists():
        return {"error": "Config file not found", "path": str(config_path)}

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)

        mcp_servers = data.get("mcpServers", {})
        if not mcp_servers:
            return {"valid": False, "error": "Missing 'mcpServers' key or empty"}

        server_count = len([k for k in mcp_servers if not k.startswith("_")])
        return {"valid": True, "server_count": server_count, "path": str(config_path)}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"JSON Syntax Error: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@server.tool()
def devtools_lint_python(file_path: str = ".") -> dict[str, Any]:
    """
    Run the 'ruff' linter on a specific file or directory.
    Returns structured JSON results of any violations found.
    """
    # Check if ruff is installed
    if not shutil.which("ruff"):
        return {"error": "Ruff is not installed or not in PATH."}

    try:
        # Run ruff check --output-format=json
        cmd = ["ruff", "check", "--output-format=json", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # If exit code is 0, no errors (usually). But ruff returns non-zero on lint errors too.
        # We parse stdout.
        output = result.stdout.strip()

        # If empty and stderr has content, something crashed or misconfigured
        if not output and result.stderr:
            # Check if it was just a "no errors" case or actual failure
            if result.returncode == 0:
                return {"success": True, "violations": []}
            return {"error": f"Ruff execution failed: {result.stderr}"}

        if not output:
            return {"success": True, "violations": []}

        try:
            violations = json.loads(output)
            return {
                "success": len(violations) == 0,
                "violation_count": len(violations),
                "violations": violations,
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse ruff JSON output", "raw_output": output}

    except Exception as e:
        return {"error": str(e)}


@server.tool()
def devtools_lint_js(file_path: str = ".") -> dict[str, Any]:
    """
    Run 'oxlint' on a specific file or directory (for JS/TS).
    Returns structured JSON results.
    """
    if not shutil.which("oxlint"):
        return {"error": "oxlint is not installed or not in PATH."}

    try:
        # oxlint --format json
        cmd = ["oxlint", "--format", "json", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        output = result.stdout.strip()
        if not output:
            return {"success": True, "violations": []}

        try:
            data = json.loads(output)
            # Oxlint JSON format is typically an array of objects
            violations = data if isinstance(data, list) else data.get("messages", [])
            return {
                "success": len(violations) == 0,
                "violation_count": len(violations),
                "violations": violations
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse oxlint JSON output", "raw_output": output}

    except Exception as e:
        return {"error": str(e)}


@server.tool()
def devtools_find_dead_code(target_path: str = ".") -> dict[str, Any]:
    """
    Run 'knip' to find unused files, dependencies, and exports.
    Requires 'knip' to be installed in the project (usually via npm).
    """
    if not shutil.which("knip") and not shutil.which("npx"):
        return {"error": "knip (or npx) is not found."}

    try:
        # We use npx knip --reporter json
        # NOTE: knip usually needs to run from the project root where package.json is.
        # target_path might be used as cwd if it's a directory.

        cwd = target_path if os.path.isdir(target_path) else "."

        cmd = ["npx", "knip", "--reporter", "json"]
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)

        output = result.stdout.strip()
        if not output:
            return {"success": True, "issues": []}

        try:
            data = json.loads(output)
            return {"success": True, "data": data}
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse knip JSON output",
                "raw_output": output,
                "stderr": result.stderr,
            }

    except Exception as e:
        return {"error": str(e)}


@server.tool()
def devtools_check_integrity(path: str = "src/") -> dict[str, Any]:
    """
    Run 'pyrefly' to check code integrity and find generic coding errors.
    """
    if not shutil.which("pyrefly"):
        return {"error": "pyrefly is not installed."}

    try:
        # Run pyrefly check
        cmd = ["pyrefly", "check", path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # Simple heuristic to extract violation count if possible
        # Pyrefly usually prints something like "Found X errors"
        import re
        error_match = re.search(r"Found (\d+) error", stdout + stderr, re.IGNORECASE)
        error_count = int(error_match.group(1)) if error_match else (0 if result.returncode == 0 else -1)

        return {
            "success": result.returncode == 0,
            "error_count": error_count,
            "stdout": stdout,
            "stderr": stderr
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    server.run()
