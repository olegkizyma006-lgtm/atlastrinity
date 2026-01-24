import email.message
import importlib.util
import json
import subprocess
import urllib.error
import urllib.request

from src.brain.mcp_preflight import (
    _parse_package_arg,
    bunx_package_exists,
    check_package_arg_for_tool,
    npm_package_exists,
    npm_registry_has_version,
    scan_mcp_config_for_package_issues,
)


class DummyCompleted:
    def __init__(self, rc=0, out="1.2.3\n", err=""):
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def test_parse_package_arg_simple():
    assert _parse_package_arg("pkg@1.2.3") == ("pkg", "1.2.3")
    assert _parse_package_arg("@scope/pkg@2025.12.18") == ("@scope/pkg", "2025.12.18")
    assert _parse_package_arg("no-version") is None


def test_npm_package_exists_success(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        return DummyCompleted(rc=0, out="1.2.3\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert npm_package_exists("pkg", "1.2.3")


def test_npm_package_exists_not_found(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        return DummyCompleted(rc=1, out="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert not npm_package_exists("pkg", "0.0.0")


def test_check_package_arg_for_tool_npx(monkeypatch):
    # Simulate registry returning version object for pkg@1.0.0 and 404 for pkg@0.0.1
    def fake_registry(req, timeout):
        url = req.get_full_url() if hasattr(req, "get_full_url") else str(req)

        class R:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        if "pkg/1.0.0" in url:
            return R()
        raise urllib.error.HTTPError(url, 404, "Not found", hdrs=email.message.Message(), fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry)
    assert check_package_arg_for_tool("pkg@1.0.0", tool_cmd="npx")
    assert not check_package_arg_for_tool("pkg@0.0.1", tool_cmd="npx")


def test_bunx_package_exists_registry(monkeypatch):
    # simulate registry returning 200 for bun package version endpoint

    def fake_registry(url, timeout):
        class R:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry)
    assert bunx_package_exists("somepkg", "1.0.0")


def test_bunx_package_not_exists(monkeypatch):
    def fake_registry(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not found", hdrs=email.message.Message(), fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry)
    assert not bunx_package_exists("otherpkg", "0.0.1")


def test_npm_registry_latest(monkeypatch):
    # Prepare fake package metadata with dist-tags.latest
    def fake_registry_meta(url, timeout):
        class R:
            status = 200

            def read(self):
                return b'{"dist-tags": {"latest": "1.2.3"}, "versions": {"1.2.3": {}}}'

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry_meta)
    assert npm_registry_has_version("somepkg", "latest")


def test_check_package_arg_for_tool_latest(monkeypatch):
    def fake_registry_meta(url, timeout):
        class R:
            status = 200

            def read(self):
                return b'{"dist-tags": {"latest": "2.0.0"}, "versions": {"2.0.0": {}}}'

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry_meta)
    assert check_package_arg_for_tool("@scope/pkg@latest", tool_cmd="npx")


def test_scan_mcp_config_for_package_issues(tmp_path, monkeypatch):
    cfg = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": ["pkg@1.0.0", "/home/user", "/tmp"],
            },
            "other": {"command": "bunx", "args": ["otherpkg@0.0.1"]},
            "disabled": {"command": "npx", "args": ["skip@1.0.0"], "disabled": True},
        },
    }
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    def fake_run(cmd, capture_output, text, timeout):
        if "pkg@1.0.0" in cmd:
            return DummyCompleted(rc=0, out="1.0.0\n")
        if "otherpkg@0.0.1" in cmd:
            return DummyCompleted(rc=1, out="")
        return DummyCompleted(rc=1, out="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Also monkeypatch registry HTTP for bunx check: return 404 for otherpkg
    class FakeHTTPError(Exception):
        pass

    def fake_registry(req, timeout):
        url = req.get_full_url() if hasattr(req, "get_full_url") else str(req)
        if "otherpkg" in url:
            raise urllib.error.HTTPError(
                url,
                404,
                "Not found",
                hdrs=email.message.Message(),
                fp=None,
            )

        class R:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_registry)

    issues = scan_mcp_config_for_package_issues(p)
    assert len(issues) == 1
    assert "otherpkg@0.0.1" in issues[0]


def test_scan_mcp_config_for_python_missing(tmp_path, monkeypatch):
    cfg = {
        "mcpServers": {
            "docker": {
                "command": "python3",
                "args": ["-c", "from mcp_server_docker import main; main()"],
            },
        },
    }
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    # Simulate module not importable via importlib and subprocess

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        "src.brain.mcp_preflight._run_cmd",
        lambda cmd, timeout=5: (1, "", "ImportError"),
    )

    issues = scan_mcp_config_for_package_issues(p)
    assert any("mcp_server_docker" in s for s in issues)


def test_scan_mcp_config_for_python_present(tmp_path, monkeypatch):
    cfg = {
        "mcpServers": {
            "docker": {
                "command": "python3",
                "args": ["-c", "from mcp_server_docker import main; main()"],
            },
        },
    }
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps(cfg))

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr("src.brain.mcp_preflight._run_cmd", lambda cmd, timeout=5: (0, "", ""))

    issues = scan_mcp_config_for_package_issues(p)
    assert not any("mcp_server_docker" in s for s in issues)
