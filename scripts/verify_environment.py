#!/usr/bin/env python3
"""
AtlasTrinity Environment Verification Script
Comprehensive check of the entire system state:
- System tools (python, node, swift, etc)
- Configuration files and Environment Variables
- Directory structure and permissions
- Service status (Redis, Postgres)
- AI Model presence (STT, TTS)
- MCP Server binary paths and availability
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import dotenv

# Colors for console output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

def print_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")

def print_pass(msg: str):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")

def print_fail(msg: str):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")

def print_warn(msg: str):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_info(msg: str):
    print(f"  {Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

# Configuration Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
VENV_PATH = PROJECT_ROOT / ".venv"

REQUIRED_TOOLS = ["python3", "npm", "node", "swift", "brew", "vibe"]
REQUIRED_DIRS = [
    "logs", "memory", "screenshots", 
    "models/tts", "models/faster-whisper", 
    "mcp", "workspace", "vibe_workspace"
]
REQUIRED_CONFIG_FILES = ["config.yaml", ".env", "mcp/config.json"]

STATUS_REPORT = {
    "passed": 0,
    "failed": 0,
    "warnings": 0
}

def check_tools():
    print_header("Checking System Tools")
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path:
            print_pass(f"Tool found: {tool} ({path})")
            STATUS_REPORT["passed"] += 1
        else:
            print_fail(f"Tool MISSING: {tool}")
            STATUS_REPORT["failed"] += 1

def check_directories():
    print_header("Checking Directory Structure")
    
    # Check Config Root
    if CONFIG_ROOT.exists():
        print_pass(f"Config Root exists: {CONFIG_ROOT}")
        STATUS_REPORT["passed"] += 1
    else:
        print_fail(f"Config Root MISSING: {CONFIG_ROOT}")
        STATUS_REPORT["failed"] += 1
        return # Cannot check children

    for subdir in REQUIRED_DIRS:
        path = CONFIG_ROOT / subdir
        if path.exists():
             # Basic permission check (writeable)
             if os.access(path, os.W_OK):
                 print_pass(f"Directory exists & writable: {subdir}")
                 STATUS_REPORT["passed"] += 1
             else:
                 print_fail(f"Directory exists but NOT WRITABLE: {subdir}")
                 STATUS_REPORT["failed"] += 1
        else:
            print_fail(f"Directory MISSING: {subdir}")
            STATUS_REPORT["failed"] += 1

def check_configs():
    print_header("Checking Configuration Files")
    
    # Load .env to check vars later
    env_path = CONFIG_ROOT / ".env"
    env_vars = {}
    if env_path.exists():
        env_vars = dotenv.dotenv_values(env_path)
    
    for filename in REQUIRED_CONFIG_FILES:
        path = CONFIG_ROOT / filename
        if path.exists():
            print_pass(f"Config file exists: {filename}")
            STATUS_REPORT["passed"] += 1
            
            # Validate YAML syntax for .yaml files
            if filename.endswith(".yaml"):
                try:
                    with open(path, 'r') as f:
                        yaml.safe_load(f)
                    print_pass(f"  Valid YAML syntax: {filename}")
                except yaml.YAMLError as e:
                    print_fail(f"  INVALID YAML syntax: {filename} - {e}")
                    STATUS_REPORT["failed"] += 1

            # Validate JSON syntax
            if filename.endswith(".json"):
                try:
                    import json
                    with open(path, 'r') as f:
                        json.load(f)
                    print_pass(f"  Valid JSON syntax: {filename}")
                except json.JSONDecodeError as e:
                    print_fail(f"  INVALID JSON syntax: {filename} - {e}")
                    STATUS_REPORT["failed"] += 1
        else:
            print_fail(f"Config file MISSING: {filename}")
            STATUS_REPORT["failed"] += 1

    # Check Critical Env Vars
    print_header("Checking Environment Variables")
    critical_vars = ["COPILOT_API_KEY"]
    optional_vars = ["GITHUB_TOKEN"]
    
    for var in critical_vars:
        if env_vars.get(var):
            print_pass(f"Variable set: {var}")
            STATUS_REPORT["passed"] += 1
        else:
            print_fail(f"Critical Variable MISSING in .env: {var}")
            STATUS_REPORT["failed"] += 1
            
    for var in optional_vars:
        if env_vars.get(var):
            print_pass(f"Variable set: {var}")
        else:
            print_warn(f"Optional Variable missing: {var}")
            STATUS_REPORT["warnings"] += 1

def check_services():
    print_header("Checking Services (Redis & Postgres)")
    
    # Check Redis
    if shutil.which("redis-cli"):
        try:
            res = subprocess.run(["redis-cli", "ping"], capture_output=True, text=True, timeout=2)
            if "PONG" in res.stdout:
                print_pass("Redis is running (PONG received)")
                STATUS_REPORT["passed"] += 1
            else:
                print_fail("Redis running but returned unexpected response")
                STATUS_REPORT["failed"] += 1
        except Exception as e:
            print_fail(f"Redis check failed: {e}")
            STATUS_REPORT["failed"] += 1
    else:
        print_warn("redis-cli not found, skipping active check")

    # Check Postgres (Basic Port Check)
    # Using python socket to avoid dependency on pg_isready if missing
    import socket
    def check_port(host, port, name):
        try:
            with socket.create_connection((host, port), timeout=2):
                print_pass(f"{name} port {port} is OPEN")
                STATUS_REPORT["passed"] += 1
                return True
        except (socket.timeout, ConnectionRefusedError):
            print_fail(f"{name} port {port} is CLOSED or Unreachable")
            STATUS_REPORT["failed"] += 1
            return False
            
    check_port("localhost", 5432, "PostgreSQL")
    check_port("localhost", 6379, "Redis (Port)")

def check_models():
    print_header("Checking AI Models")
    
    stt_path = CONFIG_ROOT / "models" / "faster-whisper"
    if stt_path.exists() and any(stt_path.iterdir()):
        # Check for model.bin or similar large files
        files = list(stt_path.rglob("*.bin"))
        if len(files) > 0:
             print_pass(f"STT Model seems present ({len(files)} bin files)")
             STATUS_REPORT["passed"] += 1
        else:
             print_warn("STT Model directory exists but no .bin files found")
             STATUS_REPORT["warnings"] += 1
    else:
         print_fail("STT Model directory missing or empty")
         STATUS_REPORT["failed"] += 1

    tts_path = CONFIG_ROOT / "models" / "tts"
    if tts_path.exists(): 
         # Simple existence check for now as structure varies
         print_pass(f"TTS Model directory exists")
         STATUS_REPORT["passed"] += 1
    else:
         print_fail("TTS Model directory missing")
         STATUS_REPORT["failed"] += 1

def check_mcp_servers():
    print_header("Checking MCP Server Binaries")
    
    # 1. macos-use (Swift)
    swift_bin = PROJECT_ROOT / "vendor" / "mcp-server-macos-use" / ".build" / "release" / "mcp-server-macos-use"
    if swift_bin.exists() and os.access(swift_bin, os.X_OK):
        print_pass("macos-use binary exists and is executable")
        STATUS_REPORT["passed"] += 1
    else:
        print_fail(f"macos-use binary MISSING or not executable at {swift_bin}")
        STATUS_REPORT["failed"] += 1

    # 2. Node MCPs (check node_modules)
    node_mcps = [
        "@modelcontextprotocol/server-filesystem",
        "@modelcontextprotocol/server-memory", 
        "@modelcontextprotocol/server-puppeteer",
        "@modelcontextprotocol/server-sequential-thinking"
    ]
    
    for mcp in node_mcps:
        path = PROJECT_ROOT / "node_modules" / mcp
        if path.exists():
            print_pass(f"Node MCP installed: {mcp}")
            STATUS_REPORT["passed"] += 1
        else:
            print_fail(f"Node MCP MISSING in node_modules: {mcp}")
            STATUS_REPORT["failed"] += 1

def main():
    print_header("ATLAS TRINITY ENVIRONMENT VERIFICATION")
    print_info(f"Project Root: {PROJECT_ROOT}")
    print_info(f"Config Root: {CONFIG_ROOT}")
    
    check_tools()
    check_directories()
    check_configs()
    check_services()
    check_models()
    check_mcp_servers()
    
    print_header("Summary")
    print(f"Passed:   {STATUS_REPORT['passed']}")
    print(f"Failed:   {STATUS_REPORT['failed']}")
    print(f"Warnings: {STATUS_REPORT['warnings']}")
    
    if STATUS_REPORT['failed'] > 0:
        print(f"\n{Colors.FAIL}Run 'python3 scripts/setup_dev.py' to attempt auto-repair.{Colors.ENDC}")
        sys.exit(1)
    else:
        print(f"\n{Colors.OKGREEN}System looks good! Ready to launch.{Colors.ENDC}")
        sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVerification cancelled.")
