import json

from scripts import check_mcp_preflight as cli


def test_run_no_issues(monkeypatch, tmp_path):
    # Create a fake config file
    cfg = {"mcpServers": {}}
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    # Monkeypatch the scanner to return no issues
    monkeypatch.setattr("src.brain.mcp_preflight.scan_mcp_config_for_package_issues", lambda x: [])
    # Also bypass system limits detection for this unit test
    monkeypatch.setattr("src.brain.mcp_preflight.check_system_limits", list)
    rc = cli.run(str(p))
    assert rc == 0


def test_run_with_issues(monkeypatch, tmp_path):
    cfg = {"mcpServers": {"filesystem": {"command": "npx", "args": ["badpkg@0.0.1"]}}}
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    monkeypatch.setattr(
        "scripts.check_mcp_preflight.scan_mcp_config_for_package_issues",
        lambda x: ["filesystem: badpkg@0.0.1 not found"],
    )
    rc = cli.run(str(p))
    assert rc == 1


def test_run_with_bunx_issues(monkeypatch, tmp_path):
    cfg = {"mcpServers": {"other": {"command": "bunx", "args": ["badbun@0.0.1"]}}}
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    monkeypatch.setattr(
        "scripts.check_mcp_preflight.scan_mcp_config_for_package_issues",
        lambda x: ["other: badbun@0.0.1 not found"],
    )
    rc = cli.run(str(p))
    assert rc == 1


def test_run_positive_both(monkeypatch, tmp_path):
    cfg = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "@modelcontextprotocol/server-filesystem@2025.12.18",
                    "${HOME}",
                    "/tmp",
                ],
            },
            "dev-bun": {"command": "bunx", "args": ["chrome-devtools-mcp@latest"]},
        },
    }
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    # Simulate no issues returned by scanner
    monkeypatch.setattr(
        "scripts.check_mcp_preflight.scan_mcp_config_for_package_issues",
        lambda x: [],
    )
    # Also bypass system limits detection for this unit test
    monkeypatch.setattr("src.brain.mcp_preflight.check_system_limits", list)
    rc = cli.run(str(p))
    assert rc == 0


def test_run_with_system_limits(monkeypatch, tmp_path):
    cfg = {"mcpServers": {}}
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    # Simulate check_system_limits returning an issue
    monkeypatch.setattr(
        "src.brain.mcp_preflight.check_system_limits",
        lambda: ["Process limit per user is low: 256 (soft)"],
    )
    # And scanner returns no package issues
    monkeypatch.setattr(
        "scripts.check_mcp_preflight.scan_mcp_config_for_package_issues",
        lambda x: [],
    )
    rc = cli.run(str(p))
    # By default system-limit issues are warnings (do not fail preflight)
    assert rc == 0
