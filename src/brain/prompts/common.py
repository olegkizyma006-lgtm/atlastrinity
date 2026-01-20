"""
Common constants and shared fragments for prompts.

This module now uses the centralized mcp_registry for dynamic catalog generation.
"""

from ..mcp_registry import (
    DATA_PROTOCOL,
    SDLC_PROTOCOL,
    SEARCH_PROTOCOL,
    STORAGE_PROTOCOL,
    SYSTEM_MASTERY_PROTOCOL,
    TASK_PROTOCOL,
    VIBE_DOCUMENTATION,
    VOICE_PROTOCOL,
    get_server_catalog_for_prompt,
)

# Re-export VOICE_PROTOCOL directly
__all__ = [
    "DATA_PROTOCOL",
    # Legacy exports for backwards compatibility
    "DEFAULT_REALM_CATALOG",
    "SDLC_PROTOCOL",
    "SEARCH_PROTOCOL",
    "STORAGE_PROTOCOL",
    "SYSTEM_MASTERY_PROTOCOL",
    "TASK_PROTOCOL",
    "VIBE_TOOLS_DOCUMENTATION",
    "VOICE_PROTOCOL",
    "get_realm_catalog",
    "get_vibe_documentation",
]


def get_realm_catalog() -> str:
    """
    Get current realm catalog, generated dynamically from mcp_registry.
    This replaces the hardcoded DEFAULT_REALM_CATALOG.
    """
    return get_server_catalog_for_prompt(include_key_tools=True)


def get_vibe_documentation() -> str:
    """
    Get Vibe tools documentation from registry.
    This replaces the hardcoded VIBE_TOOLS_DOCUMENTATION.
    """
    return VIBE_DOCUMENTATION


# ═══════════════════════════════════════════════════════════════════════════════
#                    LEGACY COMPATIBILITY (will be removed in future)
# ═══════════════════════════════════════════════════════════════════════════════
# These are kept for backwards compatibility with any code that imports them directly.
# New code should use get_realm_catalog() and get_vibe_documentation() instead.

DEFAULT_REALM_CATALOG = get_realm_catalog()
VIBE_TOOLS_DOCUMENTATION = get_vibe_documentation()
