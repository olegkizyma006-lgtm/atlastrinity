#!/usr/bin/env python3
"""
AtlasTrinity Full Stack Development Setup Script
Виконує комплексне налаштування середовища після клонування:
- Перевірка середовища (Python 3.12.12, Bun, Swift)
- Створення та синхронізація глобальних конфігурацій (~/.config/atlastrinity)
- Компіляція нативних MCP серверів (Swift)
- Встановлення Python та NPM залежностей
- Завантаження AI моделей (STT/TTS)
- Перевірка системних сервісів (Docker, Redis, Postgres)
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# Кольори для консолі
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_step(msg: str):
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}[SETUP]{Colors.ENDC} {msg}")


def print_success(msg: str):
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")


def print_error(msg: str):
    print(f"{Colors.FAIL}✗{Colors.ENDC} {msg}")


def print_info(msg: str):
    print(f"{Colors.OKCYAN}ℹ{Colors.ENDC} {msg}")


# Константи
REQUIRED_PYTHON = "3.12.12"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = Path.home() / ".config" / "atlastrinity"
VENV_PATH = PROJECT_ROOT / ".venv"

# Папки для конфігів та моделей
DIRS = {
    "config": CONFIG_ROOT,
    "logs": CONFIG_ROOT / "logs",
    "memory": CONFIG_ROOT / "memory",
    "screenshots": CONFIG_ROOT / "screenshots",
    "tts_models": CONFIG_ROOT / "models" / "tts",
    "stt_models": CONFIG_ROOT / "models" / "faster-whisper",
    "mcp": CONFIG_ROOT / "mcp",
    "workspace": Path.home() / "AtlasProjects",
}


def check_python_version():
    """Перевіряє версію Python"""
    print_step(f"Перевірка версії Python (ціль: {REQUIRED_PYTHON})...")
    current_version = platform.python_version()

    if current_version == REQUIRED_PYTHON:
        print_success(f"Python {current_version} знайдено")
        return True
    else:
        print_warning(f"Поточна версія Python: {current_version}")
        print_info(f"Рекомендовано використовувати {REQUIRED_PYTHON} для повної сумісності.")
        return True  # Дозволяємо продовжити, але з попередженням


def ensure_directories():
    """Створює необхідні директорії в ~/.config"""
    print_step("Налаштування глобальних директорій...")
    for name, path in DIRS.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Створено {name}: {path}")
        else:
            print_success(f"Директорія {name} вже існує")


def check_system_tools():
    """Перевіряє наявність базових інструментів"""
    print_step("Перевірка базових інструментів...")
    tools = ["brew", "bun", "swift", "npm", "vibe"]
    missing = []

    for tool in tools:
        path = shutil.which(tool)
        if path:
            try:
                if tool == "swift":
                    version = subprocess.check_output([tool, "--version"]).decode().splitlines()[0]
                elif tool == "vibe":
                    # Vibe might not support --version or behave differently, try standard
                    try:
                        version = (
                            subprocess.check_output([tool, "--version"], timeout=2).decode().strip()
                        )
                    except Exception:
                        version = "detected"
                else:
                    version = subprocess.check_output([tool, "--version"]).decode().strip()
                print_success(f"{tool} знайдено ({version})")
            except Exception:
                print_success(f"{tool} знайдено")
        else:
            if tool == "vibe":
                print_warning("Vibe CLI не знайдено! (Потрібен для coding tasks)")
            else:
                print_warning(f"{tool} НЕ знайдено")
            missing.append(tool)

    if "bun" in missing:
        print_info("Bun не знайдено. Встановлення Bun...")
        try:
            subprocess.run("curl -fsSL https://bun.sh/install | bash", shell=True, check=True)
            # Add to PATH for current session
            bun_bin = Path.home() / ".bun" / "bin"
            os.environ["PATH"] += os.pathsep + str(bun_bin)
            print_success("Bun встановлено")
            # Remove from missing list if successful
            missing.remove("bun")
        except Exception as e:
            print_error(f"Не вдалося встановити Bun: {e}")
    if "swift" in missing:
        print_error("Swift необхідний для компіляції macos-use MCP серверу!")

    return "brew" not in missing  # Brew є обов'язковим


def ensure_database():
    """Перевіряє наявність бази даних та створює її, якщо потрібно"""
    print_step("Налаштування бази даних PostgreSQL...")
    db_name = "atlastrinity_db"

    # 1. Спроба підключення до postgres (системної) для створення цільової БД
    try:
        # Перевіряємо чи існує база
        check_cmd = [
            "psql",
            "-U",
            "dev",
            "-d",
            "postgres",
            "-t",
            "-c",
            f"SELECT 1 FROM pg_database WHERE datname='{db_name}';",
        ]
        result = subprocess.run(check_cmd, capture_output=True, text=True)

        if "1" in result.stdout:
            print_success(f"База даних {db_name} вже існує")
        else:
            print_info(f"Створення бази даних {db_name}...")
            # Ensure role 'dev' exists, attempt to create if missing
            try:
                rc = subprocess.run(
                    ["psql", "-tAc", "SELECT 1 FROM pg_roles WHERE rolname='dev';"],
                    capture_output=True,
                    text=True,
                )
                if rc.returncode == 0 and rc.stdout.strip() != "1":
                    print_info("Роль 'dev' не знайдена — намагаємось створити...")
                    try:
                        subprocess.run(["createuser", "-s", "dev"], check=True)
                        print_success("Роль 'dev' створено")
                    except subprocess.CalledProcessError as e:
                        print_warning(f"Не вдалося створити роль 'dev': {e}")
                        print_info("Створіть роль вручну: createuser -s dev")
            except Exception:
                print_info("Не вдалось перевірити роль 'dev' (psql може бути недоступен).")

            create_cmd = ["createdb", "-U", "dev", db_name]
            subprocess.run(create_cmd, check=True)
            print_success(f"Базу даних {db_name} створено успішно")

        # 2. Ініціалізація таблиць через SQLAlchemy шім
        print_info("Ініціалізація таблиць (SQLAlchemy)...")
        venv_python = str(VENV_PATH / "bin" / "python")
        init_cmd = [
            venv_python,
            "-c",
            "import asyncio; from src.brain.db.manager import db_manager; asyncio.run(db_manager.initialize())",
        ]
        # Встановлюємо PYTHONPATH щоб знайти src
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        subprocess.run(init_cmd, cwd=PROJECT_ROOT, env=env, check=True)
        print_success("Схему бази даних ініціалізовано (таблиці створено)")

    except Exception as e:
        print_warning(f"Помилка при налаштуванні БД: {e}")
        print_info("Переконайтесь, що PostgreSQL запущений і користувач 'dev' має права superuser.")


def _brew_formula_installed(formula: str) -> bool:
    rc = subprocess.run(["brew", "list", "--formula", formula], capture_output=True)
    return rc.returncode == 0


def _brew_cask_installed(cask: str, app_name: str) -> bool:
    # 1) check brew metadata
    rc = subprocess.run(["brew", "list", "--cask", cask], capture_output=True)
    if rc.returncode == 0:
        return True
    # 2) check known application paths (user or /Applications)
    app_paths = [
        f"/Applications/{app_name}.app",
        f"{os.path.expanduser('~/Applications')}/{app_name}.app",
    ]
    for p in app_paths:
        if os.path.exists(p):
            return True
    return False


def install_brew_deps():
    """Встановлює системні залежності через Homebrew"""
    print_step("Перевірка та встановлення системних залежностей (Homebrew)...")

    if not shutil.which("brew"):
        print_error(
            'Homebrew не знайдено! Встановіть: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        )
        return False

    # Формули (CLI tools)
    formulas = {
        "postgresql@17": "pg_isready",  # PostgreSQL з перевіркою через pg_isready
        "redis": "redis-cli",  # Redis з перевіркою через redis-cli
    }

    # Casks (GUI apps)
    casks = {
        "docker": "Docker",  # Docker Desktop
        "google-chrome": "Google Chrome",  # Chrome для Puppeteer
    }

    # === Встановлення формул ===
    def _brew_formula_installed(formula: str) -> bool:
        rc = subprocess.run(["brew", "list", "--formula", formula], capture_output=True)
        return rc.returncode == 0

    for formula, check_cmd in formulas.items():
        if shutil.which(check_cmd) or _brew_formula_installed(formula):
            print_success(f"{formula} вже встановлено")
        else:
            print_info(f"Встановлення {formula}...")
            try:
                subprocess.run(["brew", "install", formula], check=True)
                print_success(f"{formula} встановлено")
            except subprocess.CalledProcessError as e:
                print_error(f"Помилка встановлення {formula}: {e}")

    # === Встановлення casks ===
    def _brew_cask_installed(cask: str, app_name: str) -> bool:
        # 1) check brew metadata
        rc = subprocess.run(["brew", "list", "--cask", cask], capture_output=True)
        if rc.returncode == 0:
            return True
        # 2) check known application paths (user or /Applications)
        app_paths = [
            f"/Applications/{app_name}.app",
            f"{os.path.expanduser('~/Applications')}/{app_name}.app",
        ]
        for p in app_paths:
            if os.path.exists(p):
                return True
        return False

    for cask, app_name in casks.items():
        if _brew_cask_installed(cask, app_name):
            print_success(f"{cask} вже встановлено (виявлено локально)")
            continue

        print_info(f"Встановлення {cask}...")
        try:
            subprocess.run(["brew", "install", "--cask", cask], check=True)
            print_success(f"{cask} встановлено")
        except subprocess.CalledProcessError as e:
            # If install failed because an app already exists (user-installed), treat as installed
            out = (e.stdout or b"" if hasattr(e, "stdout") else b"").decode(errors="ignore")
            err = (e.stderr or b"" if hasattr(e, "stderr") else b"").decode(errors="ignore")
            combined = out + "\n" + err
            if (
                "already an App" in combined
                or "There is already an App" in combined
                or "installed to" in combined
            ):
                print_warning(f"{cask}: додаток вже присутній (пропускаємо інсталяцію).")
            else:
                print_warning(f"Не вдалося встановити {cask}: {e}")

    # === Запуск сервісів ===
    print_step("Запуск сервісів (PostgreSQL, Redis)...")

    services = ["postgresql@17", "redis"]
    for service in services:
        try:
            # Ensure formula installed first for formula-backed services
            if not _brew_formula_installed(service):
                print_info(f"Формула {service} не встановлена — намагаємось встановити...")
                try:
                    subprocess.run(["brew", "install", service], check=True)
                    print_success(f"{service} встановлено")
                except subprocess.CalledProcessError as e:
                    print_warning(f"Не вдалося встановити {service}: {e}")
                    # skip attempting to start
                    continue

            # Перевіряємо статус
            result = subprocess.run(
                ["brew", "services", "info", service, "--json"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # If info failed, try to start anyway
                print_info(f"Запуск {service}...")
                subprocess.run(["brew", "services", "start", service], check=True)
                print_success(f"{service} запущено")
                continue

            if '"running":true' in result.stdout or '"running": true' in result.stdout:
                print_success(f"{service} вже запущено")
            else:
                print_info(f"Запуск {service}...")
                subprocess.run(["brew", "services", "start", service], check=True)
                print_success(f"{service} запущено")
        except Exception as e:
            print_warning(f"Не вдалося запустити {service}: {e}")

    # === Перевірка Docker ===
    if shutil.which("docker"):
        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            if result.returncode == 0:
                print_success("Docker Desktop запущено")
            else:
                print_warning("Docker Desktop встановлено, але не запущено. Запустіть вручну.")
        except Exception:
            print_warning("Docker Desktop не відповідає. Переконайтесь що він запущений.")

    return True


def build_swift_mcp():
    """Компілює Swift MCP сервер (macos-use)"""
    print_step("Компіляція нативного MCP серверу (macos-use)...")
    mcp_path = PROJECT_ROOT / "vendor" / "mcp-server-macos-use"

    if not mcp_path.exists():
        print_warning("Папка vendor/mcp-server-macos-use не знайдена. Пропускаємо.")
        return True

    if not shutil.which("swift"):
        print_error("Swift не знайдено, неможливо скомпілювати!")
        return False

    try:
        print_info("Запуск 'swift build -c release' (це може зайняти час)...")
        subprocess.run(["swift", "build", "-c", "release"], cwd=mcp_path, check=True)

        binary_path = mcp_path / ".build" / "release" / "mcp-server-macos-use"
        if binary_path.exists():
            print_success(f"Скомпільовано успішно: {binary_path}")
            return True
        else:
            print_error("Бінарний файл не знайдено після компіляції!")
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"Помилка компіляції Swift: {e}")
        return False


def check_venv():
    """Налаштовує Python virtual environment"""
    print_step("Налаштування Python venv...")
    if not VENV_PATH.exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
            print_success("Virtual environment створено")
        except Exception as e:
            print_error(f"Не вдалося створити venv: {e}")
            return False
    else:
        print_success("Venv вже існує")
    return True


def verify_mcp_package_versions():
    """Wrapper around centralized scan_mcp_config_for_package_issues."""
    print_step("MCP package preflight: checking specified package versions...")

    # We need to ensure src is in path to import local module
    sys.path.append(str(PROJECT_ROOT))
    try:
        from src.brain.mcp_preflight import (
            check_system_limits,
            scan_mcp_config_for_package_issues,
        )  # noqa: E402
    except ImportError:
        print_warning("Could not import mcp_preflight. Skipping pre-check.")
        return []

    # Prefer global config path
    mcp_config_path = CONFIG_ROOT / "mcp" / "config.json"
    if not mcp_config_path.exists():
        mcp_config_path = PROJECT_ROOT / "src" / "mcp_server" / "config.json"

    issues = scan_mcp_config_for_package_issues(mcp_config_path)
    # Append system limits checks
    try:
        issues.extend(check_system_limits())
    except Exception:
        pass
    return issues


def install_deps():
    """Встановлює всі залежності (Python, NPM, MCP)"""
    print_step("Встановлення залежностей...")

    # 1. Python
    venv_python = str(VENV_PATH / "bin" / "python")
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        print_info("PIP install -r requirements.txt...")
        subprocess.run([venv_python, "-m", "pip", "install", "-U", "pip"], capture_output=True)
        subprocess.run([venv_python, "-m", "pip", "install", "-r", str(req_file)], check=True)
        # STT deps
        subprocess.run(
            [
                venv_python,
                "-m",
                "pip",
                "install",
                "faster-whisper",
                "sounddevice",
                "soundfile",
            ],
            capture_output=True,
        )
        # Try to install optional MCP python servers (best-effort). If not available, warn but do not fail setup.
        try:
            rc = subprocess.run(
                [venv_python, "-m", "pip", "install", "mcp_server_docker"],
                capture_output=True,
            )
            if rc.returncode != 0:
                print_warning(
                    "Optional package 'mcp_server_docker' not installed automatically; install it if you need the Python Docker MCP (pip install mcp_server_docker)"
                )
        except Exception:
            print_warning(
                "Failed to check/install optional 'mcp_server_docker' package; install manually if required."
            )
        print_success("Python залежності встановлено")

    # 2. NPM & MCP
    if shutil.which("npm"):
        print_info("NPM install & MCP packages...")
        subprocess.run(["npm", "install"], cwd=PROJECT_ROOT, capture_output=True, check=True)

        mcp_packages = [
            "@modelcontextprotocol/server-slack",
            "@modelcontextprotocol/server-postgres",
            "@modelcontextprotocol/server-sequential-thinking",
            "@modelcontextprotocol/server-memory",
            "@mcpcentral/mcp-time",
            "@buggyhunter/context7-mcp",
            "chrome-devtools-mcp",
            "@modelcontextprotocol/server-filesystem",
            "apple-mcp",
            "@thelord/mcp-server-docker-npx",
        ]
        subprocess.run(
            ["npm", "install"] + mcp_packages,
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )
        print_success("NPM та MCP пакети встановлено")
    else:
        print_error("NPM не знайдено!")
        return False

    return True


def sync_configs():
    """Copies template configs to global folder if they don't exist (first-time setup only)."""
    print_step("Setting up global configurations...")

    try:
        # Initial setup: copy templates if global configs don't exist
        config_yaml_src = PROJECT_ROOT / "config" / "config.yaml.template"
        config_yaml_dst = CONFIG_ROOT / "config.yaml"
        
        mcp_json_src = PROJECT_ROOT / "src" / "mcp_server" / "config.json.template"
        mcp_json_dst = CONFIG_ROOT / "mcp" / "config.json"

        # Copy config.yaml template if not exists
        if not config_yaml_dst.exists():
            if config_yaml_src.exists():
                shutil.copy2(config_yaml_src, config_yaml_dst)
                print_success(f"Created config.yaml from template")
            else:
                # Fallback: create minimal config
                import yaml
                defaults = {
                    "agents": {
                        "atlas": {"model": "gpt-5-mini", "temperature": 0.7},
                        "tetyana": {"model": "gpt-4.1", "temperature": 0.5},
                        "grisha": {"vision_model": "gpt-4o", "temperature": 0.3},
                    },
                    "mcp": {},
                    "logging": {"level": "INFO"},
                }
                with open(config_yaml_dst, "w", encoding="utf-8") as f:
                    yaml.dump(defaults, f, allow_unicode=True)
                print_success(f"Created default config.yaml")
        else:
            print_success("config.yaml already exists in global folder")

        # Copy MCP config.json if not exists
        DIRS["mcp"].mkdir(parents=True, exist_ok=True)
        if not mcp_json_dst.exists() and mcp_json_src.exists():
            shutil.copy2(mcp_json_src, mcp_json_dst)
            print_success(f"Created mcp/config.json from template")
        else:
            print_success("mcp/config.json already exists")

        # Copy .env if not exists
        env_src = PROJECT_ROOT / ".env"
        env_dst = CONFIG_ROOT / ".env"
        if env_src.exists() and not env_dst.exists():
            shutil.copy2(env_src, env_dst)
            print_success(f"Copied .env -> {env_dst}")
        
        print_info("All configurations are in ~/.config/atlastrinity/")
        print_info("Edit configs there directly (no sync needed)")
        return True
    except Exception as e:
        print_error(f"Config setup error: {e}")
        return False


def download_models():
    """Завантажує AI моделі"""
    print_step("Завантаження моделей (може тривати довго)...")
    venv_python = str(VENV_PATH / "bin" / "python")

    # Faster-Whisper
    try:
        print_info("Завантаження Faster-Whisper large-v3-turbo...")
        cmd = [
            venv_python,
            "-c",
            "from faster_whisper import WhisperModel; "
            f"WhisperModel('large-v3-turbo', device='cpu', compute_type='int8', download_root='{DIRS['stt_models']}'); "
            "print('STT OK')",
        ]
        subprocess.run(cmd, capture_output=True, timeout=600)
        print_success("STT модель готова")
    except Exception:
        print_warning("Помилка завантаження STT (буде завантажено при старті)")

    # TTS
    try:
        print_info("Ініціалізація TTS моделей...")
        cmd = [
            venv_python,
            "-c",
            "from ukrainian_tts.tts import TTS; "
            f"TTS(cache_folder='{DIRS['tts_models']}', device='cpu'); "
            "print('TTS OK')",
        ]
        subprocess.run(cmd, capture_output=True, timeout=300)
        print_success("TTS моделі готові")
    except Exception:
        print_warning("Помилка завантаження TTS")


def check_services():
    """Перевіряє запущені сервіси"""
    print_step("Перевірка системних сервісів...")

    services = {"redis": "Redis", "postgresql@17": "PostgreSQL"}

    for service, label in services.items():
        try:
            # Check via brew services (most reliable for managed services)
            # Use manual string parsing to avoid json import dependency if missing
            res = subprocess.run(
                ["brew", "services", "info", service, "--json"],
                capture_output=True,
                text=True,
            )
            # Look for running status in JSON output
            if '"running":true' in res.stdout.replace(" ", ""):
                print_success(f"{label} запущено")
                continue

            # Fallback: check functional ping (Redis only)
            if service == "redis" and shutil.which("redis-cli"):
                if subprocess.run(["redis-cli", "ping"], capture_output=True).returncode == 0:
                    print_success(f"{label} запущено (CLI)")
                    continue

            # Fallback: check functional ping (Postgres only - if linked)
            if service == "postgresql@17" and shutil.which("pg_isready"):
                if subprocess.run(["pg_isready"], capture_output=True).returncode == 0:
                    print_success(f"{label} запущено (CLI)")
                    continue

            print_warning(f"{label} НЕ запущено. Спробуйте: brew services start {service}")

        except Exception as e:
            print_warning(f"Не вдалося перевірити {label}: {e}")

    # Docker
    try:
        if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
            print_success("Docker запущено")
        else:
            print_warning("Docker Desktop НЕ запущено (Запустіть додаток Docker)")
    except Exception:
        print_warning("Docker не знайдено")


def main():
    print(
        f"\n{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════╗{Colors.ENDC}"
    )
    print(f"{Colors.HEADER}{Colors.BOLD}║  AtlasTrinity Full Stack Dev Setup      ║{Colors.ENDC}")
    print(
        f"{Colors.HEADER}{Colors.BOLD}╚══════════════════════════════════════════╝{Colors.ENDC}\n"
    )

    check_python_version()
    ensure_directories()

    if not check_system_tools():
        print_error("Homebrew є обов'язковим! Встановіть його та спробуйте знову.")
        sys.exit(1)

    if not check_venv():
        sys.exit(1)
    install_brew_deps()  # Встановлення системних залежностей (includes ensure_database)

    # Preflight: verify MCP package versions (npx invocations)
    issues = verify_mcp_package_versions()
    if issues:
        print_warning("Detected potential MCP package issues:")
        for issue in issues:
            print_warning(f"  - {issue}")
        if os.getenv("FAIL_ON_MCP_PRECHECK") == "1":
            print_error(
                "Failing setup because FAIL_ON_MCP_PRECHECK=1 and MCP precheck found issues."
            )
            sys.exit(1)
        else:
            print_info(
                "Continuing setup despite precheck issues. Set FAIL_ON_MCP_PRECHECK=1 to fail on these errors."
            )

    if not install_deps():
        sys.exit(1)

    ensure_database()  # Now dependencies are ready

    build_swift_mcp()
    sync_configs()
    download_models()
    check_services()

    print("\n" + "=" * 60)
    print_success("✅ Налаштування завершено!")
    print_info("Кроки для початку роботи:")
    print("  1. Додайте API ключі в ~/.config/atlastrinity/.env")
    print("     - COPILOT_API_KEY (обов'язково)")
    print("     - GITHUB_TOKEN (опціонально)")
    print("  2. Запустіть систему: npm run dev")
    print("")
    print_info("Доступні MCP сервери:")
    print("  - memory: Граф знань (Atlas, Grisha, Tetyana)")
    print("  - notes: Текстові нотатки та звіти (Atlas, Grisha, Tetyana)")
    print("  - macos-use: Нативний контроль macOS + Термінал (Tetyana, Grisha)")
    print("  - vibe: Coding Agent & Self-Healing (Atlas)")
    print("  - filesystem: Файлові операції (Tetyana, Grisha)")
    print("  - sequential-thinking: Глибоке мислення (Grisha)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
