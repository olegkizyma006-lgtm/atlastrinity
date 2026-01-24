import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_setup():
    print("--- 🔍 Testing Setup Logic ---")

    # 1. Test config defaults
    from src.brain.config import CONFIG_ROOT
    from src.brain.config_loader import config

    workspace = config.get("system.workspace_path")
    repo_path = config.get("system.repository_path")

    print(f"Default Workspace: {workspace}")
    print(f"Default Repo Path: {repo_path}")

    expected_ws = str(CONFIG_ROOT / "workspace")
    if workspace != expected_ws:
        print(f"❌ Error: Unexpected default workspace: {workspace}")
    else:
        print(f"✅ Default workspace correct: {workspace}")

    # 2. Test directory creation
    from src.brain.config import ensure_dirs

    # ensure_dirs() is called on import usually, but let's call it again
    ensure_dirs()

    ws_path = Path(workspace).expanduser().absolute()
    if ws_path.exists():
        print(f"✅ Directory {ws_path} exists.")
    else:
        print(f"❌ Error: Directory {ws_path} DOES NOT exist.")

    # 3. Test vibe_server path resolution
    from src.mcp_server.vibe_server import PROJECT_ROOT

    REPOSITORY_ROOT = PROJECT_ROOT
    # _run_vibe is internal, use class if needed or just skip this one if it's dead code
    # For now, let's just use what's there if it exists or mock it
    print(f"Vibe PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Vibe REPOSITORY_ROOT: {REPOSITORY_ROOT}")

    if (
        repo_path == REPOSITORY_ROOT
        or str(
            Path(repo_path).expanduser().absolute(),
        )
        == REPOSITORY_ROOT
    ):
        print("✅ Vibe REPOSITORY_ROOT matches config.")
    else:
        print(
            f"⚠️  Vibe REPOSITORY_ROOT ({REPOSITORY_ROOT}) differs from config ({repo_path}) - check symlinks/paths.",
        )


if __name__ == "__main__":
    test_setup()
