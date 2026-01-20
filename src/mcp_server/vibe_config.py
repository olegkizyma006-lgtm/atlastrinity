"""
Vibe Configuration System

Pydantic models for type-safe configuration of Mistral Vibe CLI integration.
Supports providers, models, agents, tool permissions, and MCP integration.

Based on official Mistral Vibe documentation:
https://docs.mistral.ai/vibe/configuration/

Author: AtlasTrinity Team
Date: 2026-01-20
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("vibe_config")

# =============================================================================
# ENUMS
# =============================================================================


class ToolPermission(str, Enum):
    """Tool permission levels matching Vibe CLI."""

    ALWAYS = "always"  # Auto-approve without asking
    ASK = "ask"  # Ask for confirmation
    NEVER = "never"  # Disabled


class ApiStyle(str, Enum):
    """API styles for LLM providers."""

    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Backend(str, Enum):
    """Backend types for LLM providers."""

    MISTRAL = "mistral"
    GENERIC = "generic"
    ANTHROPIC = "anthropic"


class AgentMode(str, Enum):
    """Operational modes for Vibe agent."""

    DEFAULT = "default"  # Requires approval for tools
    PLAN = "plan"  # Read-only mode
    ACCEPT_EDITS = "accept-edits"  # Auto-approve file edits only
    AUTO_APPROVE = "auto-approve"  # Auto-approve all tools


class McpTransport(str, Enum):
    """MCP server transport types."""

    HTTP = "http"
    STREAMABLE_HTTP = "streamable-http"
    STDIO = "stdio"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class ToolConfig(BaseModel):
    """Configuration for individual tool permissions."""

    permission: ToolPermission = ToolPermission.ASK


class ProviderConfig(BaseModel):
    """LLM provider configuration.

    Example:
        [[providers]]
        name = "openrouter"
        api_base = "https://openrouter.ai/api/v1"
        api_key_env_var = "OPENROUTER_API_KEY"
        api_style = "openai"
        backend = "generic"
    """

    name: str = Field(..., description="Provider identifier for referencing")
    api_base: str = Field(..., description="Base URL for API calls")
    api_key_env_var: str = Field(..., description="Environment variable for API key")
    api_style: ApiStyle = Field(ApiStyle.OPENAI, description="API style to use")
    backend: Backend = Field(Backend.GENERIC, description="Backend implementation")

    def get_api_key(self) -> str | None:
        """Retrieve API key from environment variable."""
        return os.getenv(self.api_key_env_var)

    def is_available(self) -> bool:
        """Check if provider is available (has API key set)."""
        return bool(self.get_api_key())


class ModelConfig(BaseModel):
    """Model configuration for Vibe.

    Example:
        [[models]]
        name = "mistralai/devstral-2512:free"
        provider = "openrouter"
        alias = "devstral-openrouter"
        temperature = 0.2
        input_price = 0.0
        output_price = 0.0
    """

    name: str = Field(..., description="Model identifier in provider's API")
    provider: str = Field(..., description="Provider name to use")
    alias: str = Field(..., description="Alias for referencing in Vibe")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Sampling temperature")
    input_price: float = Field(0.0, ge=0.0, description="Price per million input tokens (USD)")
    output_price: float = Field(0.0, ge=0.0, description="Price per million output tokens (USD)")
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")


class McpServerConfig(BaseModel):
    """MCP server configuration for extending Vibe.

    Example:
        [[mcp_servers]]
        name = "fetch_server"
        transport = "stdio"
        command = "uvx"
        args = ["mcp-server-fetch"]
    """

    name: str = Field(..., description="Short name for the server")
    transport: McpTransport = Field(..., description="Transport type")
    url: str | None = Field(None, description="URL for HTTP transports")
    command: str | None = Field(None, description="Command for stdio transport")
    args: list[str] = Field(default_factory=list, description="Arguments for command")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    api_key_env: str | None = Field(None, description="Env var for API key")
    api_key_header: str | None = Field(None, description="Header name for API key")
    api_key_format: str | None = Field(
        None, description="Format for API key (e.g., 'Bearer {token}')"
    )


class AgentProfileConfig(BaseModel):
    """Custom agent profile configuration.

    Stored in ~/.vibe/agents/{name}.toml or custom agents_dir.

    Example:
        active_model = "devstral-2"
        system_prompt_id = "redteam"
        disabled_tools = ["search_replace", "write_file"]

        [tools.bash]
        permission = "always"
    """

    name: str = Field(..., description="Agent profile name")
    active_model: str | None = Field(None, description="Override active model")
    system_prompt_id: str | None = Field(None, description="Custom system prompt")
    enabled_tools: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    tools: dict[str, ToolConfig] = Field(default_factory=dict)

    @classmethod
    def load_from_file(cls, path: Path) -> AgentProfileConfig:
        """Load agent profile from TOML file."""
        if not path.exists():
            raise FileNotFoundError(f"Agent profile not found: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Add name from filename
        data["name"] = path.stem

        # Parse tool configs
        if "tools" in data:
            data["tools"] = {
                k: ToolConfig(**v) if isinstance(v, dict) else ToolConfig(permission=v)
                for k, v in data["tools"].items()
            }

        return cls(**data)


class VibeConfig(BaseModel):
    """Main Vibe configuration.

    Loaded from ~/.config/atlastrinity/vibe_config.toml or
    project-local .vibe/config.toml.
    """

    # Core settings
    active_model: str = Field("devstral-2", description="Default model alias")
    system_prompt_id: str = Field("default", description="System prompt ID")
    default_mode: AgentMode = Field(AgentMode.AUTO_APPROVE, description="Default operational mode")

    # Tool patterns (glob/regex)
    enabled_tools: list[str] = Field(
        default_factory=list, description="Tools to enable (empty=all)"
    )
    disabled_tools: list[str] = Field(default_factory=list, description="Tools to disable")

    # UI settings
    disable_welcome_banner_animation: bool = Field(True, description="Disable banner in CLI mode")
    vim_keybindings: bool = Field(False, description="Use vim keybindings")
    textual_theme: str | None = Field(None, description="Textual theme name")

    # Providers and models
    providers: list[ProviderConfig] = Field(default_factory=list)
    models: list[ModelConfig] = Field(default_factory=list)

    # Tool permission overrides
    tools: dict[str, ToolConfig] = Field(default_factory=dict)

    # MCP server integrations
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)

    # Execution limits
    max_turns: int = Field(10, ge=1, le=100, description="Default max turns")
    max_price: float | None = Field(None, ge=0.0, description="Max cost per conversation (USD)")
    timeout_s: float = Field(600.0, ge=10.0, description="Default timeout in seconds")

    # Paths (resolved at runtime)
    workspace: str | None = Field(None, description="Working directory")
    vibe_home: str | None = Field(None, description="Custom VIBE_HOME directory")
    agents_dir: str | None = Field(None, description="Custom agents directory")
    prompts_dir: str | None = Field(None, description="Custom prompts directory")

    # Cached compiled patterns
    _enabled_patterns: list[Pattern | str] = []
    _disabled_patterns: list[Pattern | str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """Compile tool patterns after initialization."""
        self._enabled_patterns = self._compile_patterns(self.enabled_tools)
        self._disabled_patterns = self._compile_patterns(self.disabled_tools)

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[Pattern[str] | str]:
        """Compile glob/regex patterns for tool matching."""
        compiled: list[Pattern[str] | str] = []
        for pattern in patterns:
            if pattern.startswith("re:"):
                # Explicit regex
                try:
                    compiled.append(re.compile(pattern[3:], re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
            elif any(c in pattern for c in "*?[]"):
                # Glob pattern - keep as string for fnmatch
                compiled.append(pattern)
            elif re.search(r"[.+(){}|^$]", pattern):
                # Heuristic: looks like regex
                try:
                    compiled.append(re.compile(pattern, re.IGNORECASE))
                except re.error:
                    # Fall back to literal match
                    compiled.append(pattern)
            else:
                # Literal match
                compiled.append(pattern)
        return compiled

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled based on patterns.

        Rules:
        1. If enabled_tools is empty, all tools are enabled by default
        2. If enabled_tools has patterns, tool must match at least one
        3. If disabled_tools has patterns, tool is disabled if it matches
        4. disabled_tools takes precedence over enabled_tools
        """
        # Check disabled first (takes precedence)
        for pattern in self._disabled_patterns:
            if self._matches_pattern(tool_name, pattern):
                return False

        # If no enabled patterns, all tools are enabled
        if not self._enabled_patterns:
            return True

        # Must match at least one enabled pattern
        for pattern in self._enabled_patterns:
            if self._matches_pattern(tool_name, pattern):
                return True

        return False

    @staticmethod
    def _matches_pattern(tool_name: str, pattern: Pattern | str) -> bool:
        """Check if tool name matches a pattern."""
        if isinstance(pattern, re.Pattern):
            return bool(pattern.fullmatch(tool_name))
        else:
            # Glob or literal
            return fnmatch.fnmatch(tool_name, pattern)

    def get_tool_permission(self, tool_name: str) -> ToolPermission:
        """Get permission level for a specific tool."""
        if tool_name in self.tools:
            return self.tools[tool_name].permission
        return ToolPermission.ASK

    def get_model_by_alias(self, alias: str) -> ModelConfig | None:
        """Find model configuration by alias."""
        for model in self.models:
            if model.alias == alias:
                return model
        return None

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Find provider configuration by name."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def get_available_models(self) -> list[ModelConfig]:
        """Get list of models with available providers."""
        available = []
        for model in self.models:
            provider = self.get_provider(model.provider)
            if provider and provider.is_available():
                available.append(model)
        return available

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        vibe_home: Path | None = None,
    ) -> VibeConfig:
        """Load configuration from TOML file.

        Search order:
        1. Explicit config_path if provided
        2. ./.vibe/config.toml (project-local)
        3. ~/.config/atlastrinity/vibe_config.toml
        4. ~/.vibe/config.toml (default Vibe location)
        5. Built-in defaults

        Args:
            config_path: Explicit path to config file
            vibe_home: Custom VIBE_HOME directory

        Returns:
            Loaded VibeConfig instance
        """
        search_paths = []

        if config_path:
            search_paths.append(config_path)

        # Project-local
        search_paths.append(Path.cwd() / ".vibe" / "config.toml")

        # AtlasTrinity config
        search_paths.append(Path.home() / ".config" / "atlastrinity" / "vibe_config.toml")

        # Default Vibe home
        effective_vibe_home = vibe_home or Path(os.getenv("VIBE_HOME", str(Path.home() / ".vibe")))
        search_paths.append(effective_vibe_home / "config.toml")

        for path in search_paths:
            if path.exists():
                logger.info(f"Loading Vibe config from: {path}")
                try:
                    return cls._load_from_file(path)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue

        # Return defaults
        logger.info("Using default Vibe configuration")
        return cls()

    @classmethod
    def _load_from_file(cls, path: Path) -> VibeConfig:
        """Load configuration from a TOML file."""
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Parse nested structures
        if "providers" in data:
            data["providers"] = [ProviderConfig(**p) for p in data["providers"]]

        if "models" in data:
            data["models"] = [ModelConfig(**m) for m in data["models"]]

        if "tools" in data:
            data["tools"] = {
                k: ToolConfig(**v) if isinstance(v, dict) else ToolConfig(permission=v)
                for k, v in data["tools"].items()
            }

        if "mcp_servers" in data:
            data["mcp_servers"] = [McpServerConfig(**s) for s in data["mcp_servers"]]

        return cls(**data)

    def to_cli_args(
        self,
        prompt: str,
        mode: AgentMode | None = None,
        model: str | None = None,
        agent: str | None = None,
        session_id: str | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
        output_format: str = "streaming",
    ) -> list[str]:
        """Build Vibe CLI arguments from configuration.

        Args:
            prompt: The prompt to send
            mode: Operational mode override
            model: Model alias override
            agent: Agent profile name
            session_id: Session to resume
            max_turns: Max conversation turns
            max_price: Max cost limit
            output_format: Output format (json/streaming/text)

        Returns:
            List of CLI arguments
        """
        args = ["-p", prompt, "--output", output_format]

        # Model selection
        effective_model = model or self.active_model
        if effective_model != "devstral-2":  # Only if non-default
            args.extend(["--model", effective_model])

        # Agent profile
        if agent:
            args.extend(["--agent", agent])

        # Mode
        effective_mode = mode or self.default_mode
        if effective_mode == AgentMode.AUTO_APPROVE:
            args.append("--auto-approve")
        elif effective_mode == AgentMode.PLAN:
            args.append("--plan")
        # Note: accept-edits is handled via tool permissions

        # Session resume
        if session_id:
            args.extend(["--session", session_id])

        # Limits
        effective_max_turns = max_turns or self.max_turns
        if effective_max_turns != 10:  # Only if non-default
            args.extend(["--max-turns", str(effective_max_turns)])

        effective_max_price = max_price or self.max_price
        if effective_max_price:
            args.extend(["--max-price", str(effective_max_price)])

        return args

    def get_environment(self) -> dict[str, str]:
        """Get environment variables for Vibe subprocess.

        Returns:
            Dictionary of environment variables to set
        """
        env = {
            "TERM": "dumb",
            "PAGER": "cat",
            "NO_COLOR": "1",
            "PYTHONUNBUFFERED": "1",
            "VIBE_DEBUG_RAW": "false",
        }

        # Custom VIBE_HOME
        if self.vibe_home:
            env["VIBE_HOME"] = self.vibe_home

        return env


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_agent_profile(
    agent_name: str,
    agents_dir: Path | None = None,
) -> AgentProfileConfig | None:
    """Load an agent profile by name.

    Args:
        agent_name: Profile name (without .toml extension)
        agents_dir: Custom agents directory

    Returns:
        AgentProfileConfig if found, None otherwise
    """
    search_dirs = []

    if agents_dir:
        search_dirs.append(agents_dir)

    # AtlasTrinity agents
    search_dirs.append(Path.home() / ".config" / "atlastrinity" / "vibe" / "agents")

    # Default Vibe agents
    vibe_home = Path(os.getenv("VIBE_HOME", str(Path.home() / ".vibe")))
    search_dirs.append(vibe_home / "agents")

    for dir_path in search_dirs:
        profile_path = dir_path / f"{agent_name}.toml"
        if profile_path.exists():
            try:
                return AgentProfileConfig.load_from_file(profile_path)
            except Exception as e:
                logger.warning(f"Failed to load agent profile {profile_path}: {e}")
                continue

    return None


def get_default_providers() -> list[ProviderConfig]:
    """Get default provider configurations."""
    return [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            api_style=ApiStyle.MISTRAL,
            backend=Backend.MISTRAL,
        ),
        ProviderConfig(
            name="openrouter",
            api_base="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER_API_KEY",
            api_style=ApiStyle.OPENAI,
            backend=Backend.GENERIC,
        ),
    ]


def get_default_models() -> list[ModelConfig]:
    """Get default model configurations."""
    return [
        ModelConfig(
            name="devstral-2",
            provider="mistral",
            alias="devstral-2",
            temperature=0.2,
            input_price=0.0,
            output_price=0.0,
        ),
        ModelConfig(
            name="mistralai/devstral-2512:free",
            provider="openrouter",
            alias="devstral-openrouter",
            temperature=0.2,
            input_price=0.0,
            output_price=0.0,
        ),
    ]
