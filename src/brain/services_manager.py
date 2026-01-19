"""
AtlasTrinity Services Manager
Handles system-level dependencies like Redis and Docker:
- Check if installed
- Install or Update if missing
- Ensure service is running
"""

import os
import shutil
import subprocess
from pathlib import Path

from .config import CONFIG_ROOT
from .logger import logger


class ServiceStatus:
    is_ready = False
    status_message = "Initializing..."
    details = {}


def is_brew_available() -> bool:
    """Check if Homebrew is installed"""
    return shutil.which("brew") is not None


def check_redis_installed() -> bool:
    """Check if redis-server is in PATH"""
    return shutil.which("redis-server") is not None


def check_docker_installed() -> bool:
    """Check if Docker Desktop is installed"""
    return os.path.exists("/Applications/Docker.app") or shutil.which("docker") is not None


def is_docker_running() -> bool:
    """Check if Docker daemon is active"""
    # docker info is a reliable way to check if daemon is responsive
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def run_command(cmd: list, timeout: int = 300) -> bool:
    """Run a system command and return success"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"[Services] Command failed {' '.join(cmd)}: {e}")
        return False


def ensure_redis(force_check: bool = False):
    """
    Ensure Redis is installed, updated, and running.
    If it's the first run (flag file missing), we attempt install/upgrade.
    """
    flag_file = CONFIG_ROOT / ".redis_ready"
    first_run = not flag_file.exists() or force_check

    if not first_run:
        # Just ensure it's running quickly
        if run_command(["redis-cli", "ping"]):
            return True
        logger.info("[Services] Redis not responding, attempting to start...")
    else:
        logger.info("[Services] Performing first-run Redis check/update...")

    if not is_brew_available():
        logger.warning("[Services] Homebrew not found. Cannot automate Redis management.")
        return False

    installed = check_redis_installed()

    if not installed:
        logger.info("[Services] Redis not found. Installing via Homebrew...")
        if run_command(["brew", "install", "redis"]):
            logger.info("[Services] ✓ Redis installed successfully.")
        else:
            logger.error("[Services] ✗ Failed to install Redis.")
            return False
    elif first_run:
        # User asked to check for updates on first start
        logger.info("[Services] Updating Redis...")
        run_command(["brew", "upgrade", "redis"])

    # Ensure service is running
    # brew services start redis is idempotent
    if run_command(["brew", "services", "start", "redis"]):
        # Verify connection
        if run_command(["redis-cli", "ping"]):
            logger.info("[Services] ✓ Redis is running and reachable.")
            if first_run:
                flag_file.touch()
            return True
        else:
            logger.warning("[Services] ! Redis service started but ping failed.")
            return False
    else:
        logger.error("[Services] ✗ Failed to start Redis service.")
        return False


def ensure_docker(force_check: bool = False):
    """
    Ensure Docker Desktop is installed and running.
    On Mac, start means launching the app.
    """
    flag_file = CONFIG_ROOT / ".docker_ready"
    first_run = not flag_file.exists() or force_check

    if not first_run:
        if is_docker_running():
            return True
        logger.info("[Services] Docker not responding, attempting to launch...")
    else:
        logger.info("[Services] Performing first-run Docker check/update...")

    if not is_brew_available():
        logger.warning("[Services] Homebrew not found. Cannot automate Docker management.")
        return False

    installed = check_docker_installed()

    if not installed:
        logger.info("[Services] Docker not found. Installing Docker Desktop via Homebrew Cask...")
        # Note: Brew might ask for password via Mac secondary dialog
        if run_command(["brew", "install", "--cask", "docker"]):
            logger.info("[Services] ✓ Docker Desktop installed.")
        else:
            logger.error("[Services] ✗ Failed to install Docker Desktop.")
            return False
    elif first_run:
        logger.info("[Services] Checking for Docker updates...")
        # Try to upgrade, ignore errors if it fails due to being already up to date
        subprocess.run(["brew", "upgrade", "--cask", "docker"], capture_output=True)

    if not is_docker_running():
        logger.info("[Services] Launching Docker Desktop app...")
        run_command(["open", "-a", "Docker"])

        # Wait for Docker to start (it can take up to 2 minutes)
        import time

        max_retries = 45  # ~90 seconds
        for i in range(max_retries):
            if is_docker_running():
                logger.info("[Services] ✓ Docker is now active.")
                if first_run:
                    flag_file.touch()
                return True
            if i % 5 == 0:
                logger.info(f"[Services] Waiting for Docker to wake up... ({i * 2}s)")
            time.sleep(2)

        logger.error("[Services] ✗ Docker failed to start within timeout.")
        return False

    if first_run:
        flag_file.touch()
    return True


def ensure_postgres(force_check: bool = False) -> bool:
    """
    Ensure PostgreSQL is installed and running.
    """
    flag_file = CONFIG_ROOT / ".postgres_ready"
    first_run = not flag_file.exists() or force_check

    if not first_run:
        # Quick check
        if shutil.which("pg_isready") and run_command(["pg_isready"]):
            return True

    logger.info("[Services] Checking PostgreSQL status...")

    # 1. Check strict availability via pg_isready
    pg_isready = shutil.which("pg_isready")
    if not pg_isready:
        # Fallback for unlinked brew postgres
        macos_brews = [
            "/opt/homebrew/opt/postgresql@17/bin/pg_isready",
            "/opt/homebrew/bin/pg_isready",
            "/usr/local/bin/pg_isready",
        ]
        for p in macos_brews:
            if os.path.exists(p):
                pg_isready = p
                break

    if pg_isready:
        if run_command([pg_isready]):
            logger.info("[Services] ✓ PostgreSQL is ready.")
            if first_run:
                flag_file.touch()
            return True
        else:
            logger.info("[Services] PostgreSQL is not responding. Attempting to start via brew...")
    else:
        logger.warning("[Services] pg_isready not found. Cannot verify connectivity precisely.")

    # 2. Try to start via Homebrew
    if is_brew_available():
        # Check if installed
        res = subprocess.run(["brew", "list", "--formula", "postgresql@17"], capture_output=True)
        if res.returncode != 0:
            logger.info("[Services] PostgreSQL@17 not installed. Installing...")
            if not run_command(["brew", "install", "postgresql@17"]):
                logger.error("[Services] Failed to install PostgreSQL.")
                return False

        # Start service
        if run_command(["brew", "services", "start", "postgresql@17"]):
            # Wait a bit for startup
            import time

            time.sleep(3)

            # Create DB if needed
            try:
                # Check if DB exists by trying to connect (using psql list) or just try create
                # Simplest is just try createdb, ignore if exists
                subprocess.run(["createdb", "atlastrinity_db"], capture_output=True)
                logger.info("[Services] Database 'atlastrinity_db' ensured.")
            except Exception as e:
                logger.warning(f"[Services] Failed to create DB: {e}")

            if shutil.which("pg_isready") and run_command(["pg_isready"]):
                logger.info("[Services] ✓ PostgreSQL started and ready.")
                if first_run:
                    flag_file.touch()
                return True
    return False


def ensure_database(force_check: bool = False) -> bool:
    """
    Ensure the configured structured database is available. Supports SQLite (default) and PostgreSQL.
    For SQLite the DB is file-based and considered available if the config root is writable.
    """
    # Lazy import to avoid circulars
    from .config_loader import config

    db_url = config.get("database.url", f"sqlite+aiosqlite:///{CONFIG_ROOT}/atlastrinity.db")
    if db_url.startswith("sqlite"):
        try:
            CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
            # Touch file to ensure directory permissions
            db_file = CONFIG_ROOT / "atlastrinity.db"
            if not db_file.exists():
                db_file.touch()
            return True
        except Exception as e:
            logger.error(f"[Services] SQLite DB not writable or cannot be created: {e}")
            return False
    else:
        # Fallback to existing postgres check for non-sqlite URLs
        return ensure_postgres(force_check=force_check)


def ensure_chrome(force_check: bool = False) -> bool:
    """
    Ensure Google Chrome is installed (required for Puppeteer execution).
    """
    # Simply check for standard paths
    paths = [
        "/Applications/Google Chrome.app",
        os.path.expanduser("~/Applications/Google Chrome.app"),
    ]

    found = any(os.path.exists(p) for p in paths)
    if found:
        # logger.info("[Services] ✓ Google Chrome detected.") # Too verbose for every run
        return True

    logger.warning("[Services] Google Chrome not found in standard paths.")

    if is_brew_available():
        logger.info("[Services] Attempting to install Google Chrome via Homebrew...")
        if run_command(["brew", "install", "--cask", "google-chrome"]):
            logger.info("[Services] ✓ Google Chrome installed.")
            return True

    return False


async def ensure_all_services(force_check: bool = False):
    """
    Run check for all required system services asynchronously.
    Updates ServiceStatus as it progresses.
    """
    import asyncio

    ServiceStatus.is_ready = False
    ServiceStatus.status_message = "Checking system services..."

    try:
        # Check Redis (blocking but fast if running)
        ServiceStatus.status_message = "Ensuring Redis is active..."
        redis_ok = await asyncio.to_thread(ensure_redis, force_check)
        ServiceStatus.details["redis"] = "ok" if redis_ok else "failed"

        # Check Docker (blocking and potentially slow)
        ServiceStatus.status_message = "Ensuring Docker is active (may take time)..."
        docker_ok = await asyncio.to_thread(ensure_docker, force_check)
        ServiceStatus.details["docker"] = "ok" if docker_ok else "failed"

        # Check Database (respects config: SQLite or PostgreSQL)
        ServiceStatus.status_message = "Checking Database..."
        db_ok = await asyncio.to_thread(ensure_database, force_check)
        ServiceStatus.details["database"] = "ok" if db_ok else "failed"

        # Check Chrome (non-blocking, just file check)
        chrome_ok = await asyncio.to_thread(ensure_chrome, force_check)
        ServiceStatus.details["chrome"] = "ok" if chrome_ok else "missing"

        if redis_ok and docker_ok and db_ok:
            ServiceStatus.is_ready = True
            ServiceStatus.status_message = "System services ready"
            logger.info("[Services] All system services are ready.")
        else:
            ServiceStatus.status_message = "Some services failed to start"
            logger.warning(f"[Services] Readiness: Redis={redis_ok}, Docker={docker_ok}, DB={db_ok}")

    except Exception as e:
        ServiceStatus.status_message = f"Service check error: {str(e)}"
        logger.error(f"[Services] Error in ensure_all_services: {e}")

    return ServiceStatus.is_ready


if __name__ == "__main__":
    # Test execution
    ensure_redis()
