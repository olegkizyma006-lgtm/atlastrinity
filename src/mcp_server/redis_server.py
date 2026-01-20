import json
import os
import sys
from typing import Any, cast

import redis.asyncio as redis
from mcp.server import FastMCP

# Setup paths for internal imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(current_dir, "..", "..")
sys.path.insert(0, os.path.abspath(root))

from src.brain.config_loader import config  # noqa: E402
from src.brain.logger import logger  # noqa: E402

server = FastMCP("redis")

# Global Redis Client
_redis_client: redis.Redis | None = None


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        # Use central config
        redis_url = (
            os.getenv("REDIS_URL") or config.get("state.redis_url") or "redis://localhost:6379/0"
        )
        try:
            _redis_client = redis.Redis.from_url(
                redis_url, decode_responses=True, socket_connect_timeout=2
            )
            # Ping is async in redis.asyncio
            # We skip ping in initialization to avoid blocking if we don't have an event loop yet
            # FastMCP will handle the async context
            logger.info("[REDIS-MCP] Connected to Redis")
        except Exception as e:
            logger.error(f"[REDIS-MCP] Failed to connect to Redis: {e}")
            raise RuntimeError(f"Could not connect to Redis: {e}")
    return _redis_client


@server.tool()
async def redis_get(key: str) -> dict[str, Any]:
    """
    Get the value of a key from Redis.
    Args:
        key: The Redis key
    """
    try:
        r = get_redis_client()
        val = await r.get(key)
        if val is None:
            return {"success": True, "key": key, "value": None, "found": False}

        # Try to parse as JSON if it looks like it
        try:
            if val.startswith("{") or val.startswith("["):
                val = json.loads(val)
        except Exception:
            pass

        return {"success": True, "key": key, "value": val, "found": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_set(key: str, value: Any, ex_seconds: int | None = None) -> dict[str, Any]:
    """
    Set the value of a key in Redis.
    Args:
        key: The Redis key
        value: The value to set (will be JSON serialized if not a string)
        ex_seconds: Optional expiry time in seconds
    """
    try:
        r = get_redis_client()
        if not isinstance(value, str):
            value = json.dumps(value, default=str)

        await r.set(key, value, ex=ex_seconds)
        return {"success": True, "key": key, "status": "OK"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_keys(pattern: str = "*") -> dict[str, Any]:
    """
    List keys matching a pattern.
    Args:
        pattern: Redis glob-style pattern (default: "*")
    """
    try:
        r = get_redis_client()
        keys = await r.keys(pattern)
        return {"success": True, "keys": list(keys), "count": len(keys)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_delete(key: str) -> dict[str, Any]:
    """
    Delete a key from Redis.
    Args:
        key: The Redis key
    """
    try:
        r = get_redis_client()
        deleted = await r.delete(key)
        return {"success": True, "key": key, "deleted": bool(deleted)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_info() -> dict[str, Any]:
    """
    Get Redis server information and statistics.
    """
    try:
        r = get_redis_client()
        info = await r.info()
        # Filter for most useful info to avoid context bloat
        essential_info = {
            "version": info.get("redis_version"),
            "uptime_days": info.get("uptime_in_days"),
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "db0_keys": info.get("db0", {}).get("keys", 0),
        }
        return {"success": True, "info": essential_info}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_ttl(key: str) -> dict[str, Any]:
    """
    Get the time-to-live (TTL) for a key in seconds.
    Args:
        key: The Redis key
    """
    try:
        r = get_redis_client()
        ttl = await r.ttl(key)
        # Redis TTL returns -1 for no expiry, -2 for not found
        return {"success": True, "key": key, "ttl": ttl}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_hgetall(key: str) -> dict[str, Any]:
    """
    Get all fields and values in a hash.
    Args:
        key: The Redis key (must be a hash)
    """
    try:
        r = get_redis_client()
        val = await cast(Any, r.hgetall(key))
        return {"success": True, "hash": val}
    except Exception as e:
        return {"success": False, "error": str(e)}


@server.tool()
async def redis_hset(key: str, mapping: dict[str, Any]) -> dict[str, Any]:
    """
    Set multiple hash fields to multiple values.
    Args:
        key: The Redis key
        mapping: Dictionary of fields and values
    """
    try:
        r = get_redis_client()
        await cast(Any, r.hset(key, mapping=mapping))
        return {"success": True, "key": key}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    try:
        server.run()
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(0)
    except BaseException as e:
        # suppressed broken pipe logic to match other servers
        def contains_broken_pipe(exc):
            if isinstance(exc, BrokenPipeError) or "Broken pipe" in str(exc):
                return True
            if hasattr(exc, "exceptions"):
                return any(contains_broken_pipe(inner) for inner in exc.exceptions)
            return False

        if contains_broken_pipe(e):
            sys.exit(0)
        raise
