#!/usr/bin/env python3
"""
AtlasTrinity Full Stack Development Setup Script
Виконує комплексне налаштування середовища після клонування:
- Перевірка середовища (Python 3.12.12, Bun, Swift)
- Створення та синхронізація глобальних конфігурацій (~/.config/atlastrinity)
- Компіляція нативних MCP серверів (Swift)
- Встановлення Python та NPM залежностей
- Завантаження AI моделей (STT/TTS)
- Перевірка системних сервісів (Docker, Redis)
"""

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


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
    "workspace": CONFIG_ROOT / "workspace",
    "vibe_workspace": CONFIG_ROOT / "vibe_workspace",
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
    tools = ["brew", "bun", "swift", "npm", "vibe", "oxlint", "knip", "ruff", "pyrefly"]
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
            # Check venv for python tools
            if tool in ["ruff", "pyrefly"]:
                venv_tool = PROJECT_ROOT / ".venv" / "bin" / tool
                if venv_tool.exists():
                    print_success(f"{tool} знайдено у .venv")
                    continue
            if tool == "vibe":
                print_warning("Vibe CLI не знайдено! (Буде встановлено нижче)")
            else:
                print_warning(f"{tool} НЕ знайдено")
            missing.append(tool)

    # Auto-install Bun if missing
    if "bun" in missing:
        print_info("Bun не знайдено. Встановлення Bun...")
        try:
            subprocess.run("curl -fsSL https://bun.sh/install | bash", shell=True, check=True)
            # Add to PATH for current session
            bun_bin = Path.home() / ".bun" / "bin"
            os.environ["PATH"] += os.pathsep + str(bun_bin)
            print_success("Bun встановлено")
            if "bun" in missing:
                missing.remove("bun")
        except Exception as e:
            print_error(f"Не вдалося встановити Bun: {e}")

    # Auto-install Vibe if missing
    if "vibe" in missing:
        print_info("Vibe CLI не знайдено. Встановлення Vibe...")
        try:
            # Official vibe installation script
            subprocess.run("curl -fsSL https://get.vibe.sh | sh", shell=True, check=True)
            print_success("Vibe CLI встановлено")
            if "vibe" in missing:
                missing.remove("vibe")
        except Exception as e:
            print_error(f"Не вдалося встановити Vibe: {e}")

    # Auto-install JS dev tools if missing
    if any(t in missing for t in ["oxlint", "knip"]):
        print_info("Деякі JS інструменти (oxlint/knip) відсутні. Спроба встановити через npm...")
        try:
            subprocess.run(["npm", "install", "-g", "oxlint", "knip"], check=False)
            print_success("JS інструменти встановлено")
            for t in ["oxlint", "knip"]:
                if t in missing:
                    missing.remove(t)
        except Exception:
            pass

    if "swift" in missing:
        print_error("Swift необхідний для компіляції macos-use MCP серверу!")

    return "brew" not in missing  # Brew є обов'язковим


def ensure_database():
    """Initialize SQLite database in global config folder"""
    print_step("Налаштування бази даних (SQLite)...")
    db_path = CONFIG_ROOT / "atlastrinity.db"

    # 1. Check if database file exists
    try:
        if db_path.exists():
            print_success(f"SQLite база даних вже існує: {db_path}")
        else:
            print_info(f"Створення нової SQLite бази: {db_path}...")

        # 2. Initialize tables via SQLAlchemy
        print_info("Ініціалізація таблиць (SQLAlchemy)...")
        venv_python = str(VENV_PATH / "bin" / "python")
        init_cmd = [venv_python, str(PROJECT_ROOT / "scripts" / "init_db.py")]
        subprocess.run(init_cmd, cwd=PROJECT_ROOT, check=True)
        print_success("Схему бази даних ініціалізовано (таблиці створено)")

    except Exception as e:
        print_warning(f"Помилка при налаштуванні БД: {e}")
        print_info("Переконайтесь, що aiosqlite встановлено: pip install aiosqlite")


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

    # Формули (CLI tools) - SQLite doesn't need server, only Redis for caching
    formulas = {
        "redis": "redis-cli",  # Redis для кешування активних сесій
    }

    # Casks (GUI apps)
    casks = {
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
    print_step("Запуск сервісів (Redis)...")

    services = ["redis"]  # SQLite doesn't need a server
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

            is_running = False
            if result.returncode == 0:
                if '"running":true' in result.stdout.replace(" ", ""):
                    is_running = True

            if is_running:
                print_success(f"{service} вже запущено")
            else:
                print_info(f"Запуск {service}...")
                # Use check=False and check output for 'already started'
                res = subprocess.run(
                    ["brew", "services", "start", service], capture_output=True, text=True
                )
                if res.returncode == 0 or "already started" in res.stderr.lower():
                    print_success(f"{service} запущено")
                else:
                    print_warning(f"Не вдалося запустити {service}: {res.stderr.strip()}")
        except Exception as e:
            print_warning(f"Не вдалося запустити {service}: {e}")

    return True


def build_swift_mcp():
    """Компілює Swift MCP сервер (macos-use)"""
    print_step("Компіляція нативного MCP серверу (macos-use)...")
    mcp_path = PROJECT_ROOT / "vendor" / "mcp-server-macos-use"

    if not mcp_path.exists():
        print_warning("Папка vendor/mcp-server-macos-use не знайдена. Пропускаємо.")
        return True

    # Force recompilation: removing existing binary check to ensure latest logic is built
    print_info("Forcing recompilation of macos-use to ensure binary integrity...")

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
        )
    except ImportError:
        print_warning("Could not import mcp_preflight. Skipping pre-check.")
        return []

    # Prefer global config path
    mcp_config_path = CONFIG_ROOT / "mcp" / "config.json"
    if not mcp_config_path.exists():
        mcp_config_path = CONFIG_ROOT / "mcp" / "config.json"
        if not mcp_config_path.exists():
            mcp_config_path = PROJECT_ROOT / "config" / "mcp_servers.json.template"

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
    venv_python = (
        str(VENV_PATH / "/bin/python")
        if (VENV_PATH / "bin" / "python").exists()
        else str(VENV_PATH / "bin" / "python")
    )
    # Actually VENV_PATH / "bin" / "python" is more standard, but I'll use what was there or improved.
    venv_python = str(VENV_PATH / "bin" / "python")

    # Update PIP first
    subprocess.run([venv_python, "-m", "pip", "install", "-U", "pip"], capture_output=True)

    # Install main requirements
    req_file = PROJECT_ROOT / "requirements.txt"
    if req_file.exists():
        print_info("PIP install -r requirements.txt...")
        subprocess.run([venv_python, "-m", "pip", "install", "-r", str(req_file)], check=True)

    # Install dev requirements if they exist (it's a dev setup)
    req_dev_file = PROJECT_ROOT / "requirements-dev.txt"
    if req_dev_file.exists():
        print_info("PIP install -r requirements-dev.txt...")
        subprocess.run([venv_python, "-m", "pip", "install", "-r", str(req_dev_file)], check=True)

    print_success("Python залежності встановлено")

    # 2. NPM & MCP
    if shutil.which("npm"):
        print_info("NPM install (from package.json)...")
        subprocess.run(["npm", "install"], cwd=PROJECT_ROOT, capture_output=True, check=True)

        # Critical MCP servers - ensure they are explicitly installed/updated
        # These are usually in package.json but we force-check them here
        mcp_packages = [
            "@modelcontextprotocol/server-sequential-thinking",
            "chrome-devtools-mcp",
            "@modelcontextprotocol/server-filesystem",
            "@modelcontextprotocol/server-puppeteer",
            "@modelcontextprotocol/server-github",
            "@modelcontextprotocol/server-memory",
            "@modelcontextprotocol/inspector",
        ]
        print_info("Updating critical MCP packages...")
        subprocess.run(
            ["npm", "install", *mcp_packages],
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
        # Force overwrite: copy templates to global configs
        config_yaml_src = PROJECT_ROOT / "config" / "config.yaml.template"
        config_yaml_dst = CONFIG_ROOT / "config.yaml"

        mcp_json_src = PROJECT_ROOT / "config" / "mcp_servers.json.template"
        mcp_json_dst = CONFIG_ROOT / "mcp" / "config.json"

        # Copy config.yaml template (Overwrite)
        if config_yaml_src.exists():
            shutil.copy2(config_yaml_src, config_yaml_dst)
            print_success("Overwrote config.yaml from template")
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
            print_success("Created default config.yaml (Template missing)")

        # Copy MCP config.json (Overwrite)
        DIRS["mcp"].mkdir(parents=True, exist_ok=True)
        if mcp_json_src.exists():
            shutil.copy2(mcp_json_src, mcp_json_dst)
            print_success("FORCED SYNC: Overwrote mcp/config.json from project template")
        else:
            print_warning("mcp_servers.json.template missing, skipped overwrite")

        # Copy .env if not exists
        env_src = PROJECT_ROOT / ".env"
        env_dst = CONFIG_ROOT / ".env"
        if env_src.exists() and not env_dst.exists():
            shutil.copy2(env_src, env_dst)
            print_success(f"Copied .env -> {env_dst}")

        # Copy behavior_config.yaml template (Overwrite)
        behavior_yaml_src = PROJECT_ROOT / "config" / "behavior_config.yaml.template"
        behavior_yaml_dst = CONFIG_ROOT / "behavior_config.yaml"
        if behavior_yaml_src.exists():
            shutil.copy2(behavior_yaml_src, behavior_yaml_dst)
            print_success("FORCED SYNC: Overwrote behavior_config.yaml from project template")
        else:
            print_warning("behavior_config.yaml.template missing in config/, skipped overwrite")

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

    # 1. Faster-Whisper: Detect model from config or template
    model_name = "large-v3"  # Default fallback
    try:
        import yaml

        config_path = CONFIG_ROOT / "config.yaml"
        template_path = PROJECT_ROOT / "config" / "config.yaml.template"
        target_path = config_path if config_path.exists() else template_path

        if target_path.exists():
            with open(target_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                # Try voice.stt.model or fallback to whisper_stt.model
                model_name = (
                    cfg.get("voice", {}).get("stt", {}).get("model")
                    or cfg.get("mcp", {}).get("whisper_stt", {}).get("model")
                    or model_name
                )
    except Exception:
        pass

    try:
        print_info(f"Завантаження Faster-Whisper {model_name}...")
        cmd = [
            venv_python,
            "-c",
            "from faster_whisper import WhisperModel; "
            f"WhisperModel('{model_name}', device='cpu', compute_type='int8', download_root='{DIRS['stt_models']}'); "
            "print('STT OK')",
        ]
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=900
        )  # Increased timeout for large models
        if res.returncode == 0:
            print_success(f"STT модель {model_name} готова")
        else:
            print_warning(f"Помилка завантаження STT: {res.stderr}")
    except Exception as e:
        print_warning(f"Помилка завантаження STT: {e}")

    # TTS
    try:
        print_info("Ініціалізація TTS моделей (з автоматичним патчингом)...")
        python_script = f"""
import os, sys
from pathlib import Path
sys.path.append(os.getcwd())
try:
    from src.brain.voice.tts import _patch_tts_config
except ImportError:
    def _patch_tts_config(d): pass

cache_dir = Path('{DIRS["tts_models"]}')
cache_dir.mkdir(parents=True, exist_ok=True)

from ukrainian_tts.tts import TTS

# 1. Спробуємо ініціалізацію з автоматичним патчингом у разі помилки
old_cwd = os.getcwd()
try:
    os.chdir(str(cache_dir))
    # TTS constructor downloads files if missing
    TTS(cache_folder='.', device='cpu')
except Exception as e:
    print(f"Initial TTS load failed (likely missing files or unpatched config): {{e}}")
    os.chdir(old_cwd)
    # Якщо завантаження відбулося, але завантаження не вдалося, спробуємо патчити
    _patch_tts_config(cache_dir)
    os.chdir(str(cache_dir))
    # Спроба №2 після патчу
    TTS(cache_folder='.', device='cpu')
finally:
    os.chdir(old_cwd)
print('TTS OK')
"""
        cmd = [venv_python, "-c", python_script]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if res.returncode == 0:
            print_success("TTS моделі готові")
        else:
            # Check if it failed specifically because of feats_stats.npz and try to provide helpful info
            if "feats_stats.npz" in res.stderr or "feats_stats.npz" in res.stdout:
                print_warning(
                    "Виявлено проблему з feats_stats.npz. Спробуйте вручну видалити папку models/tts та запустити знову."
                )
            print_warning(f"Помилка завантаження TTS: {res.stderr or res.stdout}")
    except Exception as e:
        print_warning(f"Помилка завантаження TTS: {e}")


def backup_databases():
    """Архівує SQLite базу та ChromaDB для синхронізації через Git"""
    print_step("Створення резервних копій баз даних...")

    backup_dir = PROJECT_ROOT / "backups" / "databases"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. Backup SQLite database
    sqlite_src = CONFIG_ROOT / "atlastrinity.db"
    if sqlite_src.exists():
        sqlite_dst = backup_dir / "atlastrinity.db"
        shutil.copy2(sqlite_src, sqlite_dst)
        print_success(f"SQLite база збережена: {sqlite_dst}")
    else:
        print_warning("SQLite база не знайдена, пропускаємо.")

    # 2. Backup ChromaDB (vector database)
    chroma_src = CONFIG_ROOT / "memory"
    if chroma_src.exists():
        chroma_dst = backup_dir / "memory"
        if chroma_dst.exists():
            shutil.rmtree(chroma_dst)
        shutil.copytree(chroma_src, chroma_dst)
        print_success(f"ChromaDB (векторна база) збережена: {chroma_dst}")
    else:
        print_warning("ChromaDB не знайдена, пропускаємо.")

    print_info(f"Резервні копії збережено в: {backup_dir}")
    print_info(
        "Тепер ви можете зробити: git add backups/ && git commit -m 'backup: database snapshot'"
    )


def restore_databases():
    """Відновлює бази даних з архіву"""
    print_step("Відновлення баз даних з резервних копій...")

    backup_dir = PROJECT_ROOT / "backups" / "databases"
    if not backup_dir.exists():
        print_warning("Резервні копії не знайдено. Виконайте git pull або backup_databases().")
        return

    # 1. Restore SQLite
    sqlite_src = backup_dir / "atlastrinity.db"
    if sqlite_src.exists():
        sqlite_dst = CONFIG_ROOT / "atlastrinity.db"
        shutil.copy2(sqlite_src, sqlite_dst)
        print_success(f"SQLite база відновлена: {sqlite_dst}")

    # 2. Restore ChromaDB
    chroma_src = backup_dir / "memory"
    if chroma_src.exists():
        chroma_dst = CONFIG_ROOT / "memory"
        if chroma_dst.exists():
            shutil.rmtree(chroma_dst)
        shutil.copytree(chroma_src, chroma_dst)
        print_success(f"ChromaDB відновлена: {chroma_dst}")

    print_success("Бази даних успішно відновлено!")


async def verify_database_tables():
    """Detailed verification of database tables and counts using external script"""
    print_step("Детальна перевірка таблиць бази даних...")
    venv_python = str(VENV_PATH / "bin" / "python")
    try:
        subprocess.run(
            [venv_python, str(PROJECT_ROOT / "scripts" / "verify_db_tables.py")], check=True
        )
        return True
    except Exception as e:
        print_error(f"Помилка при верифікації таблиць: {e}")
        return False


def check_services():
    """Перевіряє запущені сервіси"""
    print_step("Перевірка системних сервісів...")

    services = {"redis": "Redis"}  # SQLite is file-based, no service needed

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

            print_warning(f"{label} НЕ запущено. Спробуйте: brew services start {service}")

        except Exception as e:
            print_warning(f"Не вдалося перевірити {label}: {e}")


def run_integrity_check():
    """Runs ruff and oxlint to ensure the setup is clean"""
    print_step("Запуск перевірки цілісності коду (Integrity Check)...")
    venv_python = str(VENV_PATH / "bin" / "python")

    # Python checks
    try:
        print_info("Перевірка Python (Ruff)...")
        subprocess.run([venv_python, "-m", "ruff", "check", "."], cwd=PROJECT_ROOT, check=True)
        print_success("Python integrity OK")
    except subprocess.CalledProcessError:
        print_warning(
            "Виявлено проблеми в Python коді. Запустіть 'npm run format:write' для виправлення."
        )
    except Exception as e:
        print_warning(f"Не вдалося запустити Ruff: {e}")

    # TS/JS checks
    if shutil.which("oxlint"):
        try:
            print_info("Перевірка TS/JS (Oxlint)...")
            subprocess.run(["oxlint", "--ignore-path", ".gitignore"], cwd=PROJECT_ROOT, check=True)
            print_success("TS/JS integrity OK")
        except subprocess.CalledProcessError:
            print_warning("Виявлено проблеми в TS/JS коді.")
        except Exception as e:
            print_warning(f"Не вдалося запустити Oxlint: {e}")


def main():
    print(
        f"\n{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════╗{Colors.ENDC}"
    )
    print(f"{Colors.HEADER}{Colors.BOLD}║  AtlasTrinity Full Stack Dev Setup      ║{Colors.ENDC}")
    print(
        f"{Colors.HEADER}{Colors.BOLD}╚══════════════════════════════════════════╝{Colors.ENDC}\n"
    )

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser(description="AtlasTrinity Setup Script")
    parser.add_argument("--backup", action="store_true", help="Backup databases and exit")
    parser.add_argument("--restore", action="store_true", help="Restore databases and exit")
    args = parser.parse_args()

    if args.backup:
        backup_databases()
        return

    if args.restore:
        restore_databases()
        return

    check_python_version()
    ensure_directories()

    # Auto-restore databases if backups exist (from git clone)
    backup_dir = PROJECT_ROOT / "backups" / "databases"
    if backup_dir.exists() and not (CONFIG_ROOT / "atlastrinity.db").exists():
        print_info("Виявлено резервні копії баз даних у репозиторії...")
        restore_databases()

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

    # Sync configs BEFORE DB and tests to ensure latest templates are applied
    sync_configs()

    ensure_database()  # Now dependencies are ready and config is synced

    # Run detailed table verification
    asyncio.run(verify_database_tables())

    build_swift_mcp()

    # Ensure all binaries are executable
    print_step("Налаштування прав доступу для бінарних файлів...")
    bin_dirs = [PROJECT_ROOT / "bin", PROJECT_ROOT / "vendor"]
    for bdir in bin_dirs:
        if bdir.exists():
            for root, _, files in os.walk(bdir):
                for f in files:
                    fpath = Path(root) / f
                    # If it looks like an executable (macos-use, terminal, etc)
                    if "macos-use" in f or "vibe" in f or fpath.suffix == "":
                        try:
                            os.chmod(fpath, 0o755)
                        except Exception:
                            pass

    download_models()
    check_services()
    run_integrity_check()

    import json

    mcp_config_path = CONFIG_ROOT / "mcp" / "config.json"
    enabled_servers = []
    if mcp_config_path.exists():
        try:
            with open(mcp_config_path, encoding="utf-8") as f:
                mcp_cfg = json.load(f)
                servers = mcp_cfg.get("mcpServers", {})
                for s_name, s_info in servers.items():
                    if not s_info.get("disabled", False):
                        enabled_servers.append(s_name)
        except Exception:
            pass

    print("\n" + "=" * 60)
    print_success("✅ Налаштування завершено!")
    print_info("Кроки для початку роботи:")
    print("  1. Додайте API ключі в ~/.config/atlastrinity/.env")
    print("     - COPILOT_API_KEY (обов'язково)")
    print("     - GITHUB_TOKEN (опціонально)")
    print("  2. Запустіть систему: npm run dev")
    print("")

    if enabled_servers:
        print_info(f"Активні MCP сервери ({len(enabled_servers)}):")
        for srv in sorted(enabled_servers):
            print(f"  - {srv}")
    else:
        print_info("Доступні MCP сервери:")
        print("  - memory: Граф знань & Long-term Memory (Python)")
        print("  - macos-use: Нативний контроль macOS + Термінал (Swift)")
        print("  - vibe: Coding Agent & Self-Healing (Python)")
        print("  - filesystem: Файлові операції (Node)")
        print("  - sequential-thinking: Глибоке мислення (Node)")
        print("  - github: Офіційний GitHub MCP (Node)")

    print("  - MCP Inspector: Дебаг MCP серверів (npx @modelcontextprotocol/inspector)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
