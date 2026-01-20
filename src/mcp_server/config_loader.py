"""
MCP Config Loader
Loads MCP server configurations from config.yaml
"""

import os
import re
from pathlib import Path
from typing import Any

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


def load_config() -> dict[str, Any]:
    """Load full configuration from config.yaml and .env"""
    config_path = CONFIG_ROOT / "config.yaml"
    env_path = CONFIG_ROOT / ".env"

    # 1. Load .env into os.environ for placeholder substitution
    if env_path.exists():
        try:
            from dotenv import dotenv_values

            env_vars = dotenv_values(env_path)
            for key, value in (env_vars or {}).items():
                if value is not None and key not in os.environ:
                    os.environ[key] = value
        except ImportError:
            pass

    # 2. Load YAML
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"⚠️  Error loading {config_path}: {e}")

    return {}


def get_config_value(section: str, key: str, default: Any = None) -> Any:
    """Get a config value from any section with placeholder resolution"""
    full_config = load_config()
    section_data = full_config.get(section, {})

    # Handle nested sections if needed (e.g. system.workspace_path)
    if "." in section:
        parts = section.split(".")
        curr = full_config
        for p in parts:
            curr = curr.get(p, {})
        section_data = curr

    value = section_data.get(key, default) if isinstance(section_data, dict) else default
    return _substitute_placeholders(value)


def load_mcp_config() -> dict[str, Any]:
    """Deprecated: use load_config().get('mcp', {}) instead"""
    return load_config().get("mcp", {})
