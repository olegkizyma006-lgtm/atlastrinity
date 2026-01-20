import json
import subprocess
import sys
import shutil
import os
from typing import Any
from mcp.server import FastMCP

# Initialize FastMCP server
server = FastMCP("devtools-server")

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
                "violations": violations
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
             # OXLint might print to stderr or just exit 0
             return {"success": True, "violations": []}

        try:
            # Oxlint JSON format is typically { "filename": "...", "messages": [...] } or similar array
            # We treat it as generic JSON
            data = json.loads(output)
            return {"success": True, "data": data}
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
            return {"error": "Failed to parse knip JSON output", "raw_output": output, "stderr": result.stderr}
            
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
        # pyrefly check <path> --format json (hypothetical, or we parse text)
        # Assuming pyrefly has a way to output machine readable or we just capture text for now.
        # If pyrefly currently only outputs text, we will wrap it.
        
        cmd = ["pyrefly", "check", path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    server.run()
