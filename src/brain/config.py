import os
import platform
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*args, **kwargs):
        return False


# Project root (where the running code is)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Repository root (where the source code for self-healing lives)
# In development, it's the same as PROJECT_ROOT.
# In production, it should be configured or found via environment/config.
REPOSITORY_ROOT = Path(os.getenv("REPOSITORY_ROOT", PROJECT_ROOT))

# Platform check
IS_MACOS = platform.system() == "Darwin"
PLATFORM_NAME = platform.system()

# Load environment variables from global .env only
global_env = Path.home() / ".config" / "atlastrinity" / ".env"
if global_env.exists():
    load_dotenv(global_env)

# Disable opt-out telemetry (e.g., ChromaDB, LangChain)
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Centralized data storage for AtlasTrinity on macOS
# Following XDG standard/developer preference for ~/.config
CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
os.environ["CONFIG_ROOT"] = str(CONFIG_ROOT)

# Subdirectories
LOG_DIR = CONFIG_ROOT / "logs"
MEMORY_DIR = CONFIG_ROOT / "memory"
SCREENSHOTS_DIR = CONFIG_ROOT / "screenshots"
MODELS_DIR = CONFIG_ROOT / "models" / "tts"
WHISPER_DIR = CONFIG_ROOT / "models" / "faster-whisper"
STANZA_DIR = CONFIG_ROOT / "models" / "stanza"
NLTK_DIR = CONFIG_ROOT / "models" / "nltk"
VPN_DIR = CONFIG_ROOT / "models" / "vpn"  # For potentially other models
MCP_DIR = CONFIG_ROOT / "mcp"
WORKSPACE_DIR = CONFIG_ROOT / "workspace"
VIBE_WORKSPACE = CONFIG_ROOT / "vibe_workspace"

# Force libraries to use our global config paths
os.environ["STANZA_RESOURCES_DIR"] = str(STANZA_DIR)
os.environ["NLTK_DATA"] = str(NLTK_DIR)
os.environ["HF_HOME"] = str(CONFIG_ROOT / "models" / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(CONFIG_ROOT / "cache")


def ensure_dirs():
    """Ensure all required data directories exist and set global workspace permissions"""
    for d in [
        CONFIG_ROOT,
        LOG_DIR,
        MEMORY_DIR,
        SCREENSHOTS_DIR,
        MODELS_DIR,
        WHISPER_DIR,
        STANZA_DIR,
        NLTK_DIR,
        VPN_DIR,
        CONFIG_ROOT / "models" / "huggingface",
        CONFIG_ROOT / "cache",
        MCP_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Special handling for Workspaces: Create and set 777 permissions
    # We now also ensure the project development workspace exists
    try:
        from .config_loader import config as sys_config

        project_ws = (
            Path(sys_config.get("system.workspace_path", WORKSPACE_DIR)).expanduser().absolute()
        )
    except Exception:
        project_ws = WORKSPACE_DIR.expanduser().absolute()

    for ws in [WORKSPACE_DIR, VIBE_WORKSPACE, project_ws]:
        if not ws.exists():
            ws.mkdir(parents=True, exist_ok=True)
        try:
            # Set 777 permissions (rwxrwxrwx) to allow full access for all users/agents
            os.chmod(ws, 0o777)
        except Exception as e:
            # Don't print warning for user folders like Developer/Trinity if chmod fails
            if ws != project_ws:
                print(f"Warning: Failed to set 777 permissions on {ws.name}: {e}")


# Initialize directories on import to ensure they exist for logger/agents
ensure_dirs()


def get_log_path(name: str) -> Path:
    """Get full path for a log file"""
    return LOG_DIR / f"{name}.log"


def get_screenshot_path(filename: str) -> str:
    """Get full path for a screenshot (string for compatibility with tools)"""
    return str(SCREENSHOTS_DIR / filename)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
