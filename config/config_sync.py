#!/usr/bin/env python3
"""
AtlasTrinity Configuration Synchronization

Забезпечує двосторонню синхронізацію конфігів між:
- config/ (проект, source of truth для структури)
- ~/.config/atlastrinity/ (глобальна, runtime + user values)

Логіка синхронізації:
1. СТРУКТУРА береться з проекту (нові ключі додаються)
2. ЗНАЧЕННЯ зберігаються з глобальної папки (user customization)
3. При конфлікті - глобальні значення мають пріоритет

Використання:
    python config/config_sync.py sync      # Синхронізувати обидва напрямки
    python config/config_sync.py push      # Проект → Глобальна (додати нові ключі)
    python config/config_sync.py pull      # Глобальна → Проект (отримати user values)
    python config/config_sync.py status    # Показати різницю
"""

import fcntl
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Спробуємо імпортувати yaml
try:
    import yaml
except ImportError:
    print("⚠️  PyYAML не встановлено. Встановіть: pip install pyyaml")
    sys.exit(1)


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


# Шляхи
PROJECT_CONFIG_DIR = Path(__file__).parent
GLOBAL_CONFIG_DIR = Path.home() / ".config" / "atlastrinity"

# Файли для синхронізації (проект → глобальна)
CONFIG_FILES = {
    "config.yaml": {
        "project": PROJECT_CONFIG_DIR / "config.yaml",
        "global": GLOBAL_CONFIG_DIR / "config.yaml",
        "merge": True,  # Smart merge (зберігає user values)
    },
    "mcp/config.json": {
        "project": PROJECT_CONFIG_DIR.parent / "src" / "mcp_server" / "config.json",
        "global": GLOBAL_CONFIG_DIR / "mcp" / "config.json",
        "merge": False,  # Просте копіювання (системний конфіг)
    },
}


def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {msg}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def print_ok(msg: str):
    print(f"{Colors.GREEN}✓{Colors.ENDC} {msg}")


def print_warn(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.ENDC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.ENDC} {msg}")


def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ{Colors.ENDC} {msg}")


def deep_merge(base: Dict, overlay: Dict, path: str = "") -> Dict:
    """
    Глибоке об'єднання двох словників.
    base - структура (нові ключі)
    overlay - значення користувача (пріоритет)
    
    Результат: структура з base + значення з overlay
    """
    result = {}
    
    # Всі ключі з обох словників
    all_keys = set(base.keys()) | set(overlay.keys())
    
    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        
        if key in base and key in overlay:
            # Ключ є в обох
            if isinstance(base[key], dict) and isinstance(overlay[key], dict):
                # Обидва - словники, рекурсивно merge
                result[key] = deep_merge(base[key], overlay[key], current_path)
            else:
                # Значення користувача має пріоритет
                result[key] = overlay[key]
        elif key in overlay:
            # Тільки в overlay (user added) - зберігаємо
            result[key] = overlay[key]
        else:
            # Тільки в base (нове з проекту) - додаємо
            result[key] = base[key]
            print_info(f"Новий ключ: {current_path}")
    
    return result


def load_yaml(path: Path) -> Optional[Dict]:
    """Завантажує YAML файл"""
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print_error(f"Помилка читання {path}: {e}")
        return None


def save_yaml(path: Path, data: Dict, comment: str = None):
    """
    Зберігає YAML файл з коментарем.
    Використовує атомарний запис через тимчасовий файл для безпеки.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    fd, temp_path = tempfile.mkstemp(suffix='.yaml.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                if comment:
                    f.write(f"# {comment}\n")
                    f.write(f"# Синхронізовано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        
        # Atomic rename (safe on POSIX systems)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def load_json(path: Path) -> Optional[Dict]:
    """Завантажує JSON файл з shared lock для безпечного читання."""
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print_error(f"Помилка читання {path}: {e}")
        return None


def save_json(path: Path, data: Dict):
    """
    Зберігає JSON файл.
    Використовує атомарний запис через тимчасовий файл для безпеки.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory for atomic rename
    fd, temp_path = tempfile.mkstemp(suffix='.json.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            # Acquire exclusive lock
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=4, ensure_ascii=False)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        
        # Atomic rename (safe on POSIX systems)
        os.replace(temp_path, path)
    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def backup_file(path: Path):
    """Створює backup файлу"""
    if path.exists():
        backup_path = path.with_suffix(path.suffix + '.backup')
        shutil.copy2(path, backup_path)
        print_info(f"Backup: {backup_path.name}")


def sync_config_file(name: str, config: Dict, direction: str = "both"):
    """
    Синхронізує один конфіг файл
    
    direction:
        "push" - проект → глобальна
        "pull" - глобальна → проект
        "both" - обидва напрямки (merge)
    """
    project_path = config["project"]
    global_path = config["global"]
    do_merge = config.get("merge", True)
    
    print(f"\n{Colors.BOLD}📄 {name}{Colors.ENDC}")
    print(f"   Project: {project_path}")
    print(f"   Global:  {global_path}")
    
    # Визначаємо тип файлу
    is_yaml = name.endswith('.yaml')
    load_fn = load_yaml if is_yaml else load_json
    save_fn = save_yaml if is_yaml else save_json
    
    project_data = load_fn(project_path)
    global_data = load_fn(global_path)
    
    if project_data is None and global_data is None:
        print_error("Обидва файли відсутні!")
        return False
    
    if do_merge and project_data and global_data:
        # Smart merge
        if direction == "push":
            # Проект → Глобальна (нові ключі з проекту, значення з глобальної)
            merged = deep_merge(project_data, global_data)
            backup_file(global_path)
            if is_yaml:
                save_fn(global_path, merged, "AtlasTrinity Configuration (synced from project)")
            else:
                save_fn(global_path, merged)
            print_ok(f"Push: merged → {global_path.name}")
            
        elif direction == "pull":
            # Глобальна → Проект (значення користувача копіюються в проект)
            merged = deep_merge(project_data, global_data)
            backup_file(project_path)
            if is_yaml:
                save_fn(project_path, merged, "AtlasTrinity System Configuration")
            else:
                save_fn(project_path, merged)
            print_ok(f"Pull: merged → {project_path.name}")
            
        else:  # both
            # Обидва напрямки - merge в обидва файли
            merged = deep_merge(project_data, global_data)
            backup_file(global_path)
            backup_file(project_path)
            if is_yaml:
                save_fn(global_path, merged, "AtlasTrinity Configuration (synced)")
                save_fn(project_path, merged, "AtlasTrinity System Configuration")
            else:
                save_fn(global_path, merged)
                save_fn(project_path, merged)
            print_ok(f"Synced both directions")
    
    elif project_data and not global_data:
        # Тільки проект - копіюємо в глобальну
        global_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_path, global_path)
        print_ok(f"Copied project → global")
    
    elif global_data and not project_data:
        # Тільки глобальна - копіюємо в проект
        shutil.copy2(global_path, project_path)
        print_ok(f"Copied global → project")
    
    elif not do_merge:
        # Просте копіювання без merge
        if direction == "push" and project_data:
            global_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project_path, global_path)
            print_ok(f"Copied project → global (no merge)")
        elif direction == "pull" and global_data:
            shutil.copy2(global_path, project_path)
            print_ok(f"Copied global → project (no merge)")
    
    return True


def show_status():
    """Показує статус синхронізації"""
    print_header("📊 Config Sync Status")
    
    for name, config in CONFIG_FILES.items():
        project_path = config["project"]
        global_path = config["global"]
        
        print(f"\n{Colors.BOLD}📄 {name}{Colors.ENDC}")
        
        project_exists = project_path.exists()
        global_exists = global_path.exists()
        
        if project_exists:
            mtime = datetime.fromtimestamp(project_path.stat().st_mtime)
            print_ok(f"Project: {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print_error("Project: MISSING")
        
        if global_exists:
            mtime = datetime.fromtimestamp(global_path.stat().st_mtime)
            print_ok(f"Global:  {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            print_warn("Global:  MISSING")
        
        # Порівняння якщо обидва існують
        if project_exists and global_exists:
            is_yaml = name.endswith('.yaml')
            load_fn = load_yaml if is_yaml else load_json
            
            project_data = load_fn(project_path)
            global_data = load_fn(global_path)
            
            if project_data == global_data:
                print_ok("Status:  ✓ In sync")
            else:
                print_warn("Status:  ⚠ Different")


def sync_all(direction: str = "both"):
    """Синхронізує всі конфіг файли"""
    action = {
        "push": "Project → Global",
        "pull": "Global → Project", 
        "both": "Bidirectional Sync"
    }.get(direction, "Sync")
    
    print_header(f"🔄 {action}")
    
    for name, config in CONFIG_FILES.items():
        sync_config_file(name, config, direction)
    
    print(f"\n{Colors.GREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.GREEN}✅ Синхронізацію завершено!{Colors.ENDC}")
    print(f"{Colors.GREEN}{'='*60}{Colors.ENDC}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python config_sync.py <command>")
        print("\nCommands:")
        print("  sync   - Bidirectional sync (merge both)")
        print("  push   - Project → Global (add new keys, keep user values)")
        print("  pull   - Global → Project (get user values)")
        print("  status - Show sync status")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "sync":
        sync_all("both")
    elif command == "push":
        sync_all("push")
    elif command == "pull":
        sync_all("pull")
    elif command == "status":
        show_status()
    else:
        print_error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
