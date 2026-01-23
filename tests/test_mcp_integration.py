import asyncio
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _integration_enabled() -> bool:
    return os.getenv("TRINITY_INTEGRATION") in {"1", "true", "True"}


def _mcp_package_available() -> bool:
    try:

        return True
    except Exception:
        return False


def _selected_servers() -> set[str] | None:
    raw = os.getenv("TRINITY_MCP_SERVERS", "").strip()
    if not raw:
        return None
    return {s.strip() for s in raw.split(",") if s.strip()}


def _missing_required_env_vars(server_cfg: dict) -> list[str]:
    missing: list[str] = []
    env = server_cfg.get("env") or {}
    for _, v in env.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_name = v[2:-1]
            if not os.getenv(env_name):
                missing.append(env_name)

    args = server_cfg.get("args") or []
    if isinstance(args, list):
        for v in args:
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_name = v[2:-1]
                if not os.getenv(env_name):
                    missing.append(env_name)
    return missing


def _load_global_mcp_config() -> dict:
    cfg_path = Path.home() / ".config" / "atlastrinity" / "mcp" / "config.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set TRINITY_INTEGRATION=1 to run real MCP integration tests",
)
def test_mcp_servers_connect_and_list_tools():
    if not _mcp_package_available():
        pytest.skip("Python package 'mcp' is not installed")

    from src.brain.mcp_manager import MCPManager

    base_mgr = MCPManager()
    servers = base_mgr.config.get("mcpServers", {}) or {}
    server_names = [name for name in servers.keys() if not name.startswith("_")]

    selected = _selected_servers()
    if selected is not None:
        server_names = [n for n in server_names if n in selected]

    assert server_names, "No enabled MCP servers configured."

    async def _run() -> None:
        for name in sorted(server_names):
            print(f"\n--> Testing server: {name}")

            mgr = MCPManager()
            try:
                server_cfg = (mgr.config.get("mcpServers", {}) or {}).get(name) or {}
                timeout = float(server_cfg.get("connect_timeout", 30.0))

                missing_env = _missing_required_env_vars(server_cfg)
                if missing_env:
                    pytest.fail(
                        f"Missing required env vars for {name}: {sorted(set(missing_env))}. "
                        "Add them to ~/.config/atlastrinity/.env or export them in your shell.",
                    )

                session = await asyncio.wait_for(mgr.get_session(name), timeout=timeout + 5.0)
                if session is None:
                    pytest.fail(f"Could not connect to MCP server: {name}")

                tools = await asyncio.wait_for(mgr.list_tools(name), timeout=timeout + 10.0)
                assert isinstance(tools, list), (
                    f"Expected list of tools from {name}, got {type(tools).__name__}"
                )
                assert len(tools) > 0, f"No tools returned by {name}"
                print(f"<-- OK: {name} connected, {len(tools)} tools found.")
            finally:
                try:
                    await mgr.cleanup()
                except Exception:
                    pass

    asyncio.run(_run())
