"""
Production First-Run Setup
Копіює та СИНХРОНІЗУЄ конфігураційні файли з .app bundle в ~/.config/atlastrinity/
Викликається автоматично при старті production .app

SMART MERGE LOGIC:
- Нові ключі з bundle ДОДАЮТЬСЯ до існуючого конфігу
- Існуючі значення користувача ЗБЕРІГАЮТЬСЯ
- .env НЕ перезаписується (API ключі користувача)
- mcp/config.json ПЕРЕЗАПИСУЄТЬСЯ (системний, не user config)

Система читає ТІЛЬКИ з ~/.config/atlastrinity/
"""

import json
import shutil
import sys
from pathlib import Path

from .config import CONFIG_ROOT, MCP_DIR, MODELS_DIR, WHISPER_DIR, deep_merge

# Try to import yaml
try:
    import yaml

    # PyYAML has been moved to a separate package in some environments
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def is_production():
    """Перевіряє чи запущено з .app bundle"""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


def get_resources_path():
    """Отримує шлях до Resources/ в .app bundle"""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    elif getattr(sys, "frozen", False):
        return Path(sys.executable).parent.parent / "Resources"
    else:
        return Path(__file__).parent.parent.parent


def sync_yaml_config(src_path: Path, dst_path: Path) -> bool:
    """
    Smart merge для YAML конфігу.
    Додає нові ключі з bundle, зберігає значення користувача.
    """
    if not YAML_AVAILABLE:
        print("[Production Setup] ⚠️  PyYAML not available, copying instead of merging")
        shutil.copy2(src_path, dst_path)
        return True

    try:
        # Завантажуємо bundle config
        with open(src_path, encoding="utf-8") as f:
            bundle_config = yaml.safe_load(f) or {}

        if not dst_path.exists():
            # Файл не існує - просто копіюємо
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    bundle_config,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            print(f"[Production Setup] ✓ Created: {dst_path.name}")
            return True

        # Файл існує - merge
        with open(dst_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Backup
        backup_path = dst_path.with_suffix(".yaml.backup")
        shutil.copy2(dst_path, backup_path)

        # Merge: bundle structure + user values
        merged = deep_merge(bundle_config, user_config)

        # Save
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write("# AtlasTrinity Configuration (auto-synced)\n")
            f.write(
                f"# Last sync: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )
            yaml.dump(merged, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        print(f"[Production Setup] ✓ Merged: {dst_path.name} (backup: {backup_path.name})")
        return True

    except Exception as e:
        print(f"[Production Setup] ✗ Error merging YAML: {e}")
        return False


def sync_json_config(src_path: Path, dst_path: Path) -> bool:
    """
    Копіює JSON конфіг (MCP servers - системний, не user config).
    """
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with open(src_path, encoding="utf-8") as f:
            bundle_config = json.load(f) or {}

        if not dst_path.exists():
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(bundle_config, f, ensure_ascii=False, indent=2)
            print(f"[Production Setup] ✓ Created: {dst_path.name}")
            return True

        with open(dst_path, encoding="utf-8") as f:
            user_config = json.load(f) or {}

        merged = deep_merge(bundle_config, user_config)
        if merged != user_config:
            backup_path = dst_path.with_suffix(".json.backup")
            shutil.copy2(dst_path, backup_path)
            with open(dst_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"[Production Setup] ✓ Merged: {dst_path.name} (backup: {backup_path.name})")
        else:
            print(f"[Production Setup] ✓ Up-to-date: {dst_path.name}")

        return True
    except Exception as e:
        print(f"[Production Setup] ✗ Error copying JSON: {e}")
        return False


def copy_config_if_needed():
    """
    Синхронізує конфігураційні файли з Resources/ в ~/.config/atlastrinity/
    - config.yaml: SMART MERGE (нові ключі + user values)
    - mcp/config.json: REPLACE (системний конфіг)
    - .env: SKIP if exists (API ключі користувача)
    """
    if not is_production():
        print("[Production Setup] Skipping - running in development mode")
        return

    resources_path = get_resources_path()
    print(f"[Production Setup] Resources path: {resources_path}")

    # 1. .env - тільки якщо не існує (API ключі користувача)
    env_src = resources_path / ".env"
    env_dst = CONFIG_ROOT / ".env"
    if env_src.exists() and not env_dst.exists():
        env_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(env_src, env_dst)
        print("[Production Setup] ✓ Created: .env")
    elif env_dst.exists():
        print("[Production Setup] ✓ Preserved: .env (user API keys)")

    # 2. config.yaml - SMART MERGE
    yaml_src = resources_path / "config.yaml"
    yaml_dst = CONFIG_ROOT / "config.yaml"
    if yaml_src.exists():
        sync_yaml_config(yaml_src, yaml_dst)
    else:
        print(f"[Production Setup] ⚠️  Source not found: {yaml_src}")

    # 3. mcp/config.json - REPLACE (системний)
    mcp_src = resources_path / "mcp" / "config.json"
    mcp_dst = MCP_DIR / "config.json"
    if mcp_src.exists():
        sync_json_config(mcp_src, mcp_dst)
    else:
        print(f"[Production Setup] ⚠️  Source not found: {mcp_src}")

    # 4. vibe_config.toml - Simple copy if missing (Vibe manages its own defaults)
    vibe_src = resources_path / "mcp_server" / "templates" / "vibe_config.toml.template"
    vibe_dst = CONFIG_ROOT / "vibe_config.toml"
    if vibe_src.exists() and not vibe_dst.exists():
        vibe_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vibe_src, vibe_dst)
        print("[Production Setup] ✓ Created: vibe_config.toml")
    elif vibe_dst.exists():
        print("[Production Setup] ✓ Preserved: vibe_config.toml")

    print("[Production Setup] ✓ Config files synchronized")


def ensure_tts_models():
    """
    Перевіряє наявність TTS моделей в ~/.config/atlastrinity/models/tts/
    Якщо немає - виводить інструкції (ukrainian-tts завантажить автоматично)
    """
    required_files = ["model.pth", "config.yaml", "feats_stats.npz", "spk_xvector.ark"]
    missing = [f for f in required_files if not (MODELS_DIR / f).exists()]

    if missing:
        print("[Production Setup] ℹ️  TTS models will be downloaded automatically on first use")
        print(f"[Production Setup] Missing files: {missing}")
        print(f"[Production Setup] Target directory: {MODELS_DIR}")
    else:
        print(f"[Production Setup] ✓ TTS models present in {MODELS_DIR}")


def ensure_stt_models():
    """
    Перевіряє наявність Faster-Whisper моделей в ~/.config/atlastrinity/models/faster-whisper/
    """
    if not WHISPER_DIR.exists() or not any(WHISPER_DIR.iterdir()):
        print("[Production Setup] ℹ️  Whisper models will be downloaded automatically on first use")
        print(f"[Production Setup] Target directory: {WHISPER_DIR}")
    else:
        print(f"[Production Setup] ✓ Faster-Whisper models present in {WHISPER_DIR}")


def run_production_setup():
    """Головна функція - викликається при старті в production"""
    if not is_production():
        return

    print("\n" + "=" * 60)
    print("🔱 AtlasTrinity Production First-Run Setup")
    print("=" * 60)

    copy_config_if_needed()
    ensure_tts_models()
    ensure_stt_models()

    print("=" * 60)
    print("✅ Production setup complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_production_setup()
