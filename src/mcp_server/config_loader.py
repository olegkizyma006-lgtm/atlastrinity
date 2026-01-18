"""
MCP Config Loader
Loads MCP server configurations from config.yaml
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

import yaml

# Standard roots for resolution
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"


def _substitute_placeholders(value: Any) -> Any:
    """Substitute ${VAR} placeholders in strings."""
    if not isinstance(value, str):
        return value

    def replace_match(match):
        var_name = match.group(1)
        if var_name == "PROJECT_ROOT":
            return str(PROJECT_ROOT)
        if var_name == "CONFIG_ROOT":
            return str(CONFIG_ROOT)
        if var_name == "HOME":
            return str(Path.home())
        
        # Fallback to environment variables
        return os.getenv(var_name, match.group(0))

    return re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace_match, value)


def load_mcp_config() -> Dict[str, Any]:
    """Load MCP configuration from config.yaml"""
    config_path = CONFIG_ROOT / "config.yaml"

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config.get("mcp", {})
        except Exception as e:
            print(f"⚠️  Error loading {config_path}: {e}")

    return {}


def get_server_config(server_name: str) -> Dict[str, Any]:
    """Get configuration for a specific MCP server"""
    mcp_config = load_mcp_config()
    return mcp_config.get(server_name, {})


def get_config_value(server_name: str, key: str, default: Any = None) -> Any:
    """Get a specific config value for a server with placeholder resolution"""
    config = get_server_config(server_name)
    value = config.get(key, default)
    return _substitute_placeholders(value)
