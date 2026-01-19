import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover

    def dotenv_values(*args, **kwargs):
        return {}


import re


from .config import CONFIG_ROOT, MCP_DIR, PROJECT_ROOT, deep_merge


class SystemConfig:
    """Singleton for system configuration with synchronization logic."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._sync_configs()
            cls._instance._load_config()
        return cls._instance

    def _sync_configs(self):
        """
        Ensure global configuration directories exist.
        Config is read ONLY from global location (~/.config/atlastrinity/).
        User config values have PRIORITY over built-in defaults.
        No template syncing - user manages global config directly.
        """
        global_path = CONFIG_ROOT / "config.yaml"
        env_path = CONFIG_ROOT / ".env"
        mcp_global = MCP_DIR / "config.json"

        CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
        MCP_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Ensure global config.yaml exists (first run only)
        if not global_path.exists():
            with open(global_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._get_defaults(),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )

        # 2. Ensure global MCP config.json exists (first run only)
        if not mcp_global.exists():
            # Copy from bundled template
            template_mcp = PROJECT_ROOT / "src" / "mcp_server" / "config.json.template"
            if template_mcp.exists():
                shutil.copy2(template_mcp, mcp_global)

        # 2. Load .env secrets into process environment (do NOT rewrite config.yaml)
        if env_path.exists():
            env_vars = dotenv_values(env_path)
            for key, value in (env_vars or {}).items():
                if value is None:
                    continue
                if os.environ.get(key) != value:
                    os.environ[key] = value

    def _load_config(self):
        """Loads configuration exclusively from the global system folder."""
        config_path = CONFIG_ROOT / "config.yaml"
        if not config_path.exists():
            self._config = self._get_defaults()
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                self._config = deep_merge(self._get_defaults(), loaded)
        except Exception:
            self._config = self._get_defaults()

    def _get_defaults(self) -> Dict[str, Any]:
        """Default configuration with fallback to environment variables."""
        base_defaults = {
            "agents": {
                "atlas": {
                    "model": os.getenv("COPILOT_MODEL", "raptor-mini"),
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
                "tetyana": {
                    "model": os.getenv("COPILOT_MODEL", "gpt-4.1"),  # Execution (Main)
                    "reasoning_model": os.getenv(
                        "COPILOT_MODEL", "gpt-4.1"
                    ),  # Tool Selection (Reasoning)
                    "reflexion_model": os.getenv("COPILOT_MODEL", "gpt-4.1"),  # Self-Correction
                    "temperature": 0.5,
                    "max_tokens": 2000,
                },
                "grisha": {
                    "vision_model": os.getenv("VISION_MODEL", "gpt-4o"),
                    "temperature": 0.3,
                    "max_tokens": 1500,
                },
            },
            "orchestrator": {
                "max_recursion_depth": 5,
                "task_timeout": 300,
                # Who should announce and lead recovery when a step fails: 'atlas' or 'grisha'
                "recovery_voice_agent": "grisha",
                # If true, call Grisha to validate failed steps even when requires_verification is not set
                "validate_failed_steps_with_grisha": True,
                "subtask_timeout": 120,
            },
            "mcp": {
                "terminal": {"enabled": True},
                "filesystem": {"enabled": True},
                "macos_use": {"enabled": True},
                "vibe": {
                    "enabled": True,
                    "workspace": str(CONFIG_ROOT / "vibe_workspace")
                }
            },
            "security": {
                "dangerous_commands": ["rm -r", "mkfs"],
                "require_confirmation": True,
            },
            "voice": {
                "tts": {
                    "engine": os.getenv("TTS_ENGINE", "ukrainian-tts"),
                    "device": "mps",
                },
                "stt": {
                    "model": os.getenv("STT_MODEL", "large-v3-turbo"),
                    "language": "uk",
                },
            },
            "system": {
                "workspace_path": "${CONFIG_ROOT}/workspace",
                "repository_path": str(PROJECT_ROOT),  # Path to Trinity source code for self-healing
            },
            "database": {
                # Default to local SQLite (async via aiosqlite). Use DATABASE_URL env var to override.
                "url": os.getenv("DATABASE_URL", f"sqlite+aiosqlite:///{CONFIG_ROOT}/atlastrinity.db")
            },
            "logging": {"level": "INFO", "max_log_size": 10485760, "backup_count": 5},
        }

        template_yaml = PROJECT_ROOT / "config" / "config.yaml.template"
        if template_yaml.exists():
            try:
                with open(template_yaml, "r", encoding="utf-8") as f:
                    template = yaml.safe_load(f) or {}
                return deep_merge(base_defaults, template)
            except Exception:
                return base_defaults

        return base_defaults

    def _substitute_placeholders(self, value: Any) -> Any:
        """Substitute ${VAR} placeholders recursively in strings, lists, or dicts."""
        if isinstance(value, str):
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
        
        if isinstance(value, list):
            return [self._substitute_placeholders(item) for item in value]
        
        if isinstance(value, dict):
            return {k: self._substitute_placeholders(v) for k, v in value.items()}
            
        return value

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return self._substitute_placeholders(value)

    def get_api_key(self, key_name: str) -> str:
        env_map = {
            "copilot_api_key": "COPILOT_API_KEY",
            "github_token": "GITHUB_TOKEN",
            "openai_api_key": "OPENAI_API_KEY",
        }

        env_var = env_map.get(key_name)
        if env_var:
            val = os.getenv(env_var)
            if val:
                return val

        return self.get(f"api.{key_name}", "")

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Returns specific agent configuration."""
        return self.get(f"agents.{agent_name}", {})

    def get_security_config(self) -> Dict[str, Any]:
        """Returns security configuration."""
        return self.get("security", {})

    @property
    def all(self) -> Dict[str, Any]:
        return self._config


config = SystemConfig()
