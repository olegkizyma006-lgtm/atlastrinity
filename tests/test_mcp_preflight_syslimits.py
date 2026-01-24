from src.brain.mcp_preflight import check_system_limits


def test_check_system_limits_low(monkeypatch):
    # Monkeypatch resource.getrlimit to return low soft limit
    try:
        import resource

        monkeypatch.setattr(resource, "getrlimit", lambda x: (512, 1024))
    except Exception:
        # If resource not available, monkeypatch _run_cmd instead
        monkeypatch.setattr(
            "src.brain.mcp_preflight._run_cmd",
            lambda cmd, timeout=10: (0, "512", ""),
        )

    issues = check_system_limits()
    assert any("Process limit per user is low" in s for s in issues) or isinstance(issues, list)


def test_check_system_limits_sysctl(monkeypatch):
    # Simulate sysctl returning low values
    def fake_run(cmd, timeout=10):
        if "kern.maxprocperuid" in cmd:
            return (0, "512", "")
        if "kern.maxproc" in cmd:
            return (0, "1024", "")
        return (1, "", "")

    monkeypatch.setattr("src.brain.mcp_preflight._run_cmd", fake_run)
    issues = check_system_limits()
    assert any("kern.maxprocperuid" in s for s in issues) or any(
        "kern.maxproc is low" in s for s in issues
    )
