"""
Internal System Actions
Registry of functions that can be called by the Workflow Engine.
These bridge the gap between YAML configuration and Python system logic.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from src.brain.db.manager import db_manager
from src.brain.logger import logger
from src.brain.services_manager import ensure_all_services
from src.brain.state_manager import state_manager

# Action registry
_INTERNAL_ACTIONS: dict[str, Callable] = {}


def register_action(name: str):
    """Decorator to register an internal action."""

    def decorator(func: Callable):
        _INTERNAL_ACTIONS[name] = func
        return func

    return decorator


def get_action(name: str) -> Callable | None:
    """Retrieve a registered action by name."""
    return _INTERNAL_ACTIONS.get(name)


# --- Standard Actions ---


@register_action("internal.log")
async def log_action(context: dict, msg: str, level: str = "info"):
    """Log a message directly to the system logger."""
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"[WORKFLOW] {msg}")

    # Also log to Orchestrator state if available in context
    orchestrator = context.get("orchestrator")
    if orchestrator:
        await orchestrator._log(msg, source="workflow", type=level)


@register_action("internal.check_services")
async def check_services_action(context: dict, timeout: int = 60):
    """Ensure all dependent services (Redis, etc.) are running."""
    logger.info("[WORKFLOW] Checking services...")
    await ensure_all_services()
    logger.info("[WORKFLOW] Services checked.")


@register_action("internal.state_init")
async def state_init_action(context: dict, reset: bool = False):
    """Initialize or reset system state."""
    orchestrator = context.get("orchestrator")
    if orchestrator:
        if reset:
            await orchestrator.reset_session()
        # Basic init logic from original restore flow
        elif state_manager.available:
            # This mimics the specialized logic in orchestrator:initialize
            # In a full migration, this would be more granular
            pass
    logger.info("[WORKFLOW] State initialized.")


@register_action("internal.db_init")
async def db_init_action(context: dict):
    """Initialize database connection."""
    logger.info("[WORKFLOW] Initializing database...")
    if db_manager:
        await db_manager.initialize()
    logger.info("[WORKFLOW] Database initialized.")


@register_action("internal.memory_warmup")
async def memory_warmup_action(context: dict, async_warmup: bool = True):
    """Warm up memory systems/TTS if needed."""
    logger.info("[WORKFLOW] Warming up memory/voice...")
    # Add actual warmup logic here if extracted from orchestrator
