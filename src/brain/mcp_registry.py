"""
MCP Registry - Unified Tool Descriptions and Schemas

Single source of truth for all MCP server definitions, tool schemas,
and documentation. Used by all agents (Atlas, Tetyana, Grisha).

Note: This file loads definitions from JSON files in src/brain/data/
to prevent language server (Pyrefly) stack overflow crashes.
"""

from typing import Any, Dict, List, Optional
import os
import json

# Paths to data files
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "mcp_catalog.json")
SCHEMAS_PATH = os.path.join(DATA_DIR, "tool_schemas.json")
VIBE_DOCS_PATH = os.path.join(DATA_DIR, "vibe_docs.txt")
VOICE_PROTOCOL_PATH = os.path.join(DATA_DIR, "voice_protocol.txt")

# Global variables to store loaded data
SERVER_CATALOG: Dict[str, Dict[str, Any]] = {}
TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}
VIBE_DOCUMENTATION: str = ""
VOICE_PROTOCOL: str = ""

def load_registry():
    """Load registry data from JSON and text files."""
    global SERVER_CATALOG, TOOL_SCHEMAS, VIBE_DOCUMENTATION, VOICE_PROTOCOL
    
    try:
        # Load Catalog
        if os.path.exists(CATALOG_PATH):
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                SERVER_CATALOG = json.load(f)
        else:
            print(f"[Here be Dragons] Warning: Catalog file not found at {CATALOG_PATH}")
            
        # Load Schemas
        if os.path.exists(SCHEMAS_PATH):
            with open(SCHEMAS_PATH, "r", encoding="utf-8") as f:
                TOOL_SCHEMAS = json.load(f)
        else:
            print(f"[Here be Dragons] Warning: Schemas file not found at {SCHEMAS_PATH}")
            
        # Load Vibe Docs
        if os.path.exists(VIBE_DOCS_PATH):
            with open(VIBE_DOCS_PATH, "r", encoding="utf-8") as f:
                VIBE_DOCUMENTATION = f.read()
        else:
            VIBE_DOCUMENTATION = "Vibe documentation not found."
            
        # Load Voice Protocol
        if os.path.exists(VOICE_PROTOCOL_PATH):
            with open(VOICE_PROTOCOL_PATH, "r", encoding="utf-8") as f:
                VOICE_PROTOCOL = f.read()
        else:
            VOICE_PROTOCOL = "Voice protocol not found."
            
    except Exception as e:
        print(f"[Here be Dragons] Error loading MCP registry: {e}")

# Load data immediately on import
load_registry()


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
