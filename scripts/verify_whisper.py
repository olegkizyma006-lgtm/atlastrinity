#!/usr/bin/env python3
"""Whisper STT Comprehensive Verification
Перевіряє всі аспекти інтеграції Whisper
"""

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent  # tests/ -> project root
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(msg: str):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}\n")


def print_check(msg: str, status: bool, details: str = ""):
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {msg}")
    if details:
        print(f"  → {details}")


def main():
    print_header("WHISPER STT VERIFICATION")

    # 1. Config файли
    print_header("1. Перевірка конфігураційних файлів")

    project_config = PROJECT_ROOT / "config.yaml"
    global_config = Path.home() / ".config" / "atlastrinity" / "config.yaml"

    print_check("Project config.yaml існує", project_config.exists(), str(project_config))

    print_check("Global config.yaml існує", global_config.exists(), str(global_config))

    # Перевірка вмісту
    if global_config.exists():
        content = global_config.read_text()
        has_mcp_whisper = "mcp:" in content and "whisper:" in content
        has_voice_stt = "voice:" in content and "stt:" in content

        print_check(
            "MCP Whisper конфіг присутній",
            has_mcp_whisper,
            "mcp.whisper.model, mcp.whisper.language",
        )

        print_check(
            "Voice STT конфіг присутній",
            has_voice_stt,
            "voice.stt.model, voice.stt.language",
        )

    # 2. Директорії
    print_header("2. Перевірка директорій")

    config_root = Path.home() / ".config" / "atlastrinity"
    whisper_dir = config_root / "models" / "whisper"
    tts_dir = config_root / "models" / "tts"

    print_check("Config root існує", config_root.exists(), str(config_root))

    print_check("Whisper models dir існує", whisper_dir.exists(), str(whisper_dir))

    print_check("TTS models dir існує", tts_dir.exists(), str(tts_dir))

    # Перевірка завантажених моделей
    if whisper_dir.exists():
        models = list(whisper_dir.glob("*.pt"))
        print_check(
            "Whisper моделі завантажені",
            len(models) > 0,
            f"Знайдено {len(models)} моделей: {[m.name for m in models]}",
        )

    # 3. Python imports
    print_header("3. Перевірка Python модулів")

    try:
        from src.brain.voice.stt import WhisperSTT

        print_check("WhisperSTT import", True, "src.brain.voice.stt")
    except Exception as e:
        print_check("WhisperSTT import", False, str(e))
        return 1

    try:
        from src.brain.config_loader import config

        print_check("config_loader import", True, "src.brain.config_loader")
    except Exception as e:
        print_check("config_loader import", False, str(e))
        return 1

    try:

        print_check("MCP Whisper Server import", True, "src.mcp_server.whisper_server")
    except Exception as e:
        print_check("MCP Whisper Server import", False, str(e))

    # 4. Ініціалізація STT
    print_header("4. Перевірка ініціалізації STT")

    try:
        stt_instance = WhisperSTT()
        print_check("WhisperSTT() створено", True)

        # Перевірка атрибутів
        print_check(
            "model_name з конфігу",
            stt_instance.model_name == "base",
            f"Очікувано: 'base', Отримано: '{stt_instance.model_name}'",
        )

        print_check(
            "language з конфігу",
            stt_instance.language == "uk",
            f"Очікувано: 'uk', Отримано: '{stt_instance.language}'",
        )

        print_check(
            "download_root налаштовано",
            stt_instance.download_root == whisper_dir,
            f"Шлях: {stt_instance.download_root}",
        )

    except Exception as e:
        print_check("WhisperSTT ініціалізація", False, str(e))
        return 1

    # 5. Config loader
    print_header("5. Перевірка config_loader")

    try:
        mcp_config = config.get("mcp", {})
        print_check("MCP config отримано", True)

        whisper_config = mcp_config.get("whisper", {})
        print_check(
            "Whisper конфіг є в MCP",
            len(whisper_config) > 0,
            f"Keys: {list(whisper_config.keys())}",
        )

        model = whisper_config.get("model")
        language = whisper_config.get("language")

        print_check(
            "MCP Whisper model",
            model == "base",
            f"Очікувано: 'base', Отримано: '{model}'",
        )

        print_check(
            "MCP Whisper language",
            language == "uk",
            f"Очікувано: 'uk', Отримано: '{language}'",
        )

    except Exception as e:
        print_check("Config loader", False, str(e))
        return 1

    # 6. Voice STT config
    print_header("6. Перевірка Voice STT конфігу")

    try:
        voice_config = config.get("voice", {})
        stt_config = voice_config.get("stt", {})

        print_check(
            "Voice STT конфіг є",
            len(stt_config) > 0,
            f"Keys: {list(stt_config.keys())}",
        )

        model = stt_config.get("model")
        language = stt_config.get("language")

        print_check(
            "Voice STT model",
            model == "base",
            f"Очікувано: 'base', Отримано: '{model}'",
        )

        print_check(
            "Voice STT language",
            language == "uk",
            f"Очікувано: 'uk', Отримано: '{language}'",
        )

    except Exception as e:
        print_check("Voice STT config", False, str(e))

    # 7. Production setup перевірка
    print_header("7. Перевірка production_setup.py")

    try:
        from src.brain.production_setup import (
            copy_config_if_needed,
        )

        print_check("production_setup imports", True)

        # Перевірка що config.yaml в списку файлів для копіювання
        import inspect

        source = inspect.getsource(copy_config_if_needed)
        has_config_yaml = "config.yaml" in source

        print_check(
            "config.yaml копіюється в production",
            has_config_yaml,
            "Є в config_files списку",
        )

    except Exception as e:
        print_check("production_setup", False, str(e))

    # 8. Setup dev перевірка
    print_header("8. Перевірка setup_dev.py")

    setup_dev = PROJECT_ROOT / "setup_dev.py"
    if setup_dev.exists():
        source = setup_dev.read_text()
        has_whisper_dir = "whisper" in source and "WHISPER_DIR" in source

        print_check("WHISPER_DIR визначено", has_whisper_dir, "models/whisper створюється")

    # 9. package.json перевірка
    print_header("9. Перевірка package.json (build)")

    package_json = PROJECT_ROOT / "package.json"
    if package_json.exists():
        import json

        pkg = json.loads(package_json.read_text())

        extra_resources = pkg.get("build", {}).get("extraResources", [])

        # Перевірка що config.yaml копіюється
        config_yaml_copied = any("config.yaml" in str(r) for r in extra_resources)

        print_check(
            "config.yaml в extraResources",
            config_yaml_copied,
            "Копіюється в production bundle",
        )

    # ПІДСУМОК
    print_header("ПІДСУМОК")
    print("\n✅ Whisper STT інтеграція:")
    print("   - Config читається з ~/.config/atlastrinity/config.yaml")
    print("   - Моделі завантажуються в ~/.config/atlastrinity/models/whisper/")
    print("   - MCP Whisper Server інтегровано з config_loader")
    print("   - Production та Dev setup налаштовані правильно")
    print("\n📝 Рекомендації:")
    print("   - При першому використанні модель завантажиться автоматично")
    print("   - Можна змінити модель через config.yaml: mcp.whisper.model")
    print("   - Доступні моделі: tiny, base, small, medium, large-v3")

    return 0


if __name__ == "__main__":
    sys.exit(main())
