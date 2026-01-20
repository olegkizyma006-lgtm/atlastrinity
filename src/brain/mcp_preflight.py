import shutil
import subprocess
from pathlib import Path


def _run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout or "", res.stderr or ""
    except Exception as e:
        return 1, "", str(e)


def _parse_package_arg(arg: str) -> tuple[str, str] | None:
    """Parse strings like 'pkg@1.2.3' or '@scope/pkg@1.2.3' and return (pkg, ver).
    Returns None if no explicit version present or arg is a file/path.
    """
    if not isinstance(arg, str):
        return None
    if arg.startswith("@/"):
        return None
    if "@" not in arg:
        return None
    parts = arg.rsplit("@", 1)
    if len(parts) != 2:
        return None
    pkg, ver = parts[0], parts[1]
    if not pkg or not ver:
        return None
    return pkg, ver


def npm_package_exists(pkg: str, ver: str) -> bool:
    """Check if npm has pkg@ver available."""
    npm = shutil.which("npm") or "npm"
    cmd = [npm, "view", f"{pkg}@{ver}", "version"]
    rc, out, _err = _run_cmd(cmd)
    return rc == 0 and bool(out.strip())


import urllib.error  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402


def npm_registry_has_version(pkg: str, ver: str) -> bool:
    """Query the npm registry directly for pkg@ver. Returns True if exists.

    Special handling for tags (eg. 'latest'): fetch package metadata and inspect
    `dist-tags` and `versions` to confirm availability.
    """
    try:
        encoded = urllib.parse.quote(pkg, safe="")
        # If version is the literal 'latest' or looks like a non-version tag, query metadata
        if ver == "latest" or not ver[0].isdigit():
            url = f"https://registry.npmjs.org/{encoded}"
            req = urllib.request.Request(
                url, headers={"Accept": "application/vnd.npm.install-v1+json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                import json

                meta = json.loads(data.decode())
                # Check dist-tags and versions
                dist = meta.get("dist-tags", {}) or {}
                versions = meta.get("versions", {}) or {}
                if ver in dist:
                    # ensure the referenced version exists in versions
                    if dist[ver] in versions:
                        return True
                    # sometimes dist-tags may include a tag pointing to a non-published version, treat as False
                    return False
                # fallback: if 'latest' requested and versions has entries, accept
                if ver == "latest" and "latest" in dist:
                    return dist["latest"] in versions
                return False
        else:
            # Direct version lookup is faster if we know the exact version string
            url = f"https://registry.npmjs.org/{encoded}/{ver}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return getattr(resp, "status", 200) == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        return False
    except Exception:
        return False


def bunx_package_exists(pkg: str, ver: str) -> bool:
    """Check bunx packages. Prefer bun registry query if possible, otherwise fall back to npm registry."""
    # If bun is installed we could attempt bun-specific calls in future, but
    # the npm registry is authoritative and works for bunx too.
    return npm_registry_has_version(pkg, ver)


import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402


def python_module_importable(module: str) -> bool:
    """Return True if the given python module can be imported in the current environment.

    We prefer using importlib to avoid spawning subprocesses; as a fallback we try
    to run a minimal `python -c 'import <module>'` using the available python binary.
    """
    try:
        if importlib.util.find_spec(module) is not None:
            return True
    except Exception:
        pass
    # Fallback: try invoking python interpreter
    py = shutil.which("python3") or shutil.which("python") or sys.executable or "python"
    rc, _out, _err = _run_cmd([py, "-c", f"import {module}"], timeout=5)
    return rc == 0


def _extract_modules_from_python_code(code: str) -> list[str]:
    """Rudimentary parser to extract top-level module names from a python code snippet.

    Captures patterns like `from pkg.subpkg import something` and `import pkg, pkg2`.
    """
    mods: list[str] = []
    # from <module> import ...
    for m in re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)\b", code):
        mods.append(m)
    # import <module>[, <module2>]
    for imp_stmt in re.findall(r"\bimport\s+([a-zA-Z0-9_\.,\s]+)", code):
        parts = [p.strip() for p in imp_stmt.split(",") if p.strip()]
        for p in parts:
            # keep only the top-level module
            mods.append(p.split(".")[0])
    # dedupe while preserving order
    seen = set()
    out = []
    for m in mods:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def check_package_arg_for_tool(arg: str, tool_cmd: str = "npx") -> bool:
    parsed = _parse_package_arg(arg)
    if not parsed:
        return True
    pkg, ver = parsed
    # Use registry-based checks for both npm and bunx-backed packages
    if tool_cmd in ("npx", "bunx"):
        return npm_registry_has_version(pkg, ver)
    # Unknown tool: assume present
    return True


def check_system_limits() -> list[str]:
    """Check OS process limits and return list of human-readable issues found.

    Uses resource.getrlimit(RLIMIT_NPROC) where available and sysctl values on macOS
    to detect if per-user or global process limits are dangerously low.
    """
    issues: list[str] = []
    try:
        import resource

        try:
            soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
            if soft != resource.RLIM_INFINITY and soft < 1024:
                issues.append(
                    f"Process limit per user is low: {soft} (soft). Consider increasing ulimit -u)"
                )
        except Exception:
            # Not available on all platforms
            pass

        # macOS / BSD: check kern.maxproc and kern.maxprocperuid
        try:
            rc, out, err = _run_cmd(["sysctl", "-n", "kern.maxproc"])
            if rc == 0 and out.strip():
                val = int(out.strip())
                if val < 2048:
                    issues.append(
                        f"kern.maxproc is low: {val} (global max procs). Consider increasing via sysctl)"
                    )
        except Exception:
            pass
        try:
            rc, out, _err = _run_cmd(["sysctl", "-n", "kern.maxprocperuid"])
            if rc == 0 and out.strip():
                val = int(out.strip())
                if val < 1024:
                    issues.append(
                        f"kern.maxprocperuid is low: {val} (per-UID max). Consider increasing via sysctl)"
                    )
        except Exception:
            pass
    except Exception:
        # Best-effort; nothing to do if environment doesn't support checks
        pass

    return issues


def scan_mcp_config_for_package_issues(config_path: Path) -> list[str]:
    """Given a path to MCP config JSON, return list of issue strings found."""
    import json

    issues: list[str] = []
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        servers = cfg.get("mcpServers", {})
        for name, s in servers.items():
            if name.startswith("_") or s.get("disabled"):
                continue
            cmd = s.get("command")
            args = s.get("args", [])
            if not cmd or not args:
                continue
            first = args[0]
            # npm / bun checks (package@version syntax)
            if cmd in {"npx", "bunx"}:
                ok = check_package_arg_for_tool(first, tool_cmd=cmd)
                if not ok:
                    issues.append(f"{name}: package {first} not found (command={cmd})")
                continue
            # Python entrypoints: detect -m or -c usages and attempt to ensure modules are importable
            # Accept commands like 'python', 'python3', or full path to a python binary
            basename = Path(cmd).name if cmd else cmd
            if basename and basename.startswith("python"):
                # -m <module>
                if "-m" in args:
                    try:
                        idx = args.index("-m")
                        module = args[idx + 1]
                        if not python_module_importable(module):
                            issues.append(
                                f"{name}: python module {module} not importable (command={cmd} -m)"
                            )
                    except Exception:
                        issues.append(f"{name}: could not parse -m module (command={cmd})")
                # -c '<code>'
                if "-c" in args:
                    try:
                        idx = args.index("-c")
                        code = args[idx + 1]
                        mods = _extract_modules_from_python_code(code)
                        for m in mods:
                            if not python_module_importable(m):
                                issues.append(
                                    f"{name}: python module {m} not importable (command={cmd} -c)"
                                )
                    except Exception:
                        issues.append(f"{name}: could not parse -c code (command={cmd})")
    except Exception as e:
        issues.append(f"Could not read/scan MCP config: {e}")
    return issues
