"""AtlasTrinity Session Path Manager

Centralized management of session-specific paths and workspace isolation.
Each session has a dedicated folder in vibe_workspace/sessions/{date}_{short_uuid}/

Key concepts:
- Session folders: Isolated directories for each work session
- Global folder: Shared resources accessible across all sessions
- Project linking: Ability to connect existing projects to sessions

Author: AtlasTrinity Team
Date: 2026-01-24
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import CONFIG_ROOT
from .logger import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


class SessionPathManager:
    """Manages session-specific paths and workspace isolation.

    Directory structure:
    ~/.config/atlastrinity/
    ├── vibe_workspace/
    │   ├── sessions/              # Session folders
    │   │   ├── 2026-01-24_abc123/
    │   │   │   └── my_project/
    │   │   └── 2026-01-25_def456/
    │   │       └── another_project/
    │   └── global/                # Global files (outside sessions)
    │       └── shared_utilities/
    ├── workspace/                 # General workspace (non-Vibe)
    │   ├── downloads/
    │   └── vault/
    """

    def __init__(self):
        self.vibe_workspace = CONFIG_ROOT / "vibe_workspace"
        self.sessions_root = self.vibe_workspace / "sessions"
        self.global_folder = self.vibe_workspace / "global"
        self.workspace = CONFIG_ROOT / "workspace"

        # Ensure base directories exist
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create base directory structure if not exists."""
        for d in [self.sessions_root, self.global_folder, self.workspace]:
            d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[SESSION_MANAGER] Directories ensured at {self.vibe_workspace}")

    def _generate_folder_name(self, session_id: str, session_name: str | None = None) -> str:
        """Generate folder name using hybrid pattern: {date}_{short_uuid}.

        Args:
            session_id: Full UUID of the session
            session_name: Optional human-readable name (unused in hybrid pattern but stored in DB)

        Returns:
            Folder name string like '2026-01-24_abc123'
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        # Use first 8 characters of UUID for short id
        try:
            short_id = str(uuid.UUID(session_id)).split("-")[0][:8]
        except (ValueError, IndexError):
            # Fallback if session_id is not a valid UUID
            short_id = session_id[:8] if len(session_id) >= 8 else session_id
        return f"{date_str}_{short_id}"

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for filesystem use."""
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r"[^\w\-]", "_", name.lower())
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Trim underscores from ends
        return sanitized.strip("_")[:100]

    def get_or_create_session_folder(
        self,
        session_id: str,
        session_name: str | None = None,
    ) -> Path:
        """Get or create a session folder.

        If a folder for this session_id already exists, return it.
        Otherwise, create a new folder with the hybrid naming pattern.

        Args:
            session_id: UUID of the session
            session_name: Optional human-readable name for the session

        Returns:
            Path to the session folder
        """
        # First, check if we already have a folder for this session
        existing = self.get_session_folder(session_id)
        if existing:
            return existing

        # Create new folder with hybrid naming
        folder_name = self._generate_folder_name(session_id, session_name)
        folder_path = self.sessions_root / folder_name

        # Handle potential conflicts (rare, but possible)
        counter = 0
        original_name = folder_name
        while folder_path.exists():
            counter += 1
            folder_name = f"{original_name}_{counter}"
            folder_path = self.sessions_root / folder_name

        folder_path.mkdir(parents=True, exist_ok=True)

        # Create a manifest file linking folder to session_id
        manifest = folder_path / ".session_manifest.json"
        import json

        manifest.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "session_name": session_name,
                    "created_at": datetime.now().isoformat(),
                    "folder_name": folder_name,
                },
                indent=2,
            ),
        )

        logger.info(f"[SESSION_MANAGER] Created session folder: {folder_path}")
        return folder_path

    def get_session_folder(self, session_id: str) -> Path | None:
        """Find existing session folder by session_id.

        Searches through session folders for the manifest file that matches.

        Args:
            session_id: UUID of the session

        Returns:
            Path to the session folder if found, None otherwise
        """
        import json

        if not self.sessions_root.exists():
            return None

        for folder in self.sessions_root.iterdir():
            if not folder.is_dir():
                continue

            manifest = folder / ".session_manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text())
                    if data.get("session_id") == session_id:
                        return folder
                except (json.JSONDecodeError, OSError):
                    continue

        return None

    def get_global_folder(self) -> Path:
        """Get the global folder for cross-session resources.

        Returns:
            Path to the global folder
        """
        self.global_folder.mkdir(parents=True, exist_ok=True)
        return self.global_folder

    def get_workspace(self) -> Path:
        """Get the general workspace folder (non-Vibe).

        Returns:
            Path to the workspace folder
        """
        self.workspace.mkdir(parents=True, exist_ok=True)
        return self.workspace

    def find_session_by_path(self, path: str | Path) -> str | None:
        """Determine which session a file path belongs to.

        Args:
            path: File path to check

        Returns:
            Session ID if path is within a session folder, None otherwise
        """
        import json

        path = Path(path).resolve()

        # Check if path is within sessions_root
        try:
            relative = path.relative_to(self.sessions_root)
        except ValueError:
            return None

        # Get the session folder (first component of relative path)
        parts = relative.parts
        if not parts:
            return None

        session_folder = self.sessions_root / parts[0]
        manifest = session_folder / ".session_manifest.json"

        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
                return data.get("session_id")
            except (json.JSONDecodeError, OSError):
                pass

        return None

    def find_session_by_name(self, name: str) -> str | None:
        """Search for a session by project name or session name.

        Searches both manifest session_name and project folder names.

        Args:
            name: Project or session name to search

        Returns:
            Session ID if found, None otherwise
        """
        import json

        name_lower = name.lower()
        sanitized = self._sanitize_name(name)

        if not self.sessions_root.exists():
            return None

        for folder in self.sessions_root.iterdir():
            if not folder.is_dir():
                continue

            manifest = folder / ".session_manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text())

                    # Check session_name
                    session_name = data.get("session_name", "")
                    if session_name and name_lower in session_name.lower():
                        return data.get("session_id")

                    # Check project folders within session
                    for project_folder in folder.iterdir():
                        if project_folder.is_dir() and not project_folder.name.startswith("."):
                            if (
                                name_lower in project_folder.name.lower()
                                or sanitized in project_folder.name.lower()
                            ):
                                return data.get("session_id")

                except (json.JSONDecodeError, OSError):
                    continue

        return None

    def link_project_to_session(
        self,
        project_path: str | Path,
        session_id: str,
    ) -> Path | None:
        """Create a symlink to an existing project within the session folder.

        Used when user requests to work with an existing project in current session.

        Args:
            project_path: Path to the existing project
            session_id: Session ID to link the project to

        Returns:
            Path to the symlink if successful, None otherwise
        """
        project_path = Path(project_path).resolve()
        if not project_path.exists():
            logger.error(f"[SESSION_MANAGER] Project not found: {project_path}")
            return None

        session_folder = self.get_or_create_session_folder(session_id)
        link_path = session_folder / project_path.name

        # Handle name conflicts
        counter = 0
        original_name = project_path.name
        while link_path.exists():
            counter += 1
            link_path = session_folder / f"{original_name}_{counter}"

        try:
            link_path.symlink_to(project_path)
            logger.info(f"[SESSION_MANAGER] Linked {project_path} -> {link_path}")
            return link_path
        except OSError as e:
            logger.error(f"[SESSION_MANAGER] Failed to create symlink: {e}")
            return None

    def is_global_path(self, path: str | Path) -> bool:
        """Check if a path is within the global folder.

        Args:
            path: Path to check

        Returns:
            True if path is in global folder, False otherwise
        """
        path = Path(path).resolve()
        try:
            path.relative_to(self.global_folder)
            return True
        except ValueError:
            return False

    def is_session_path(self, path: str | Path) -> bool:
        """Check if a path is within any session folder.

        Args:
            path: Path to check

        Returns:
            True if path is in a session folder, False otherwise
        """
        return self.find_session_by_path(path) is not None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions with their metadata.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session info dicts with id, name, folder, created_at
        """
        import json

        sessions = []

        if not self.sessions_root.exists():
            return sessions

        for folder in sorted(self.sessions_root.iterdir(), reverse=True):
            if not folder.is_dir():
                continue

            manifest = folder / ".session_manifest.json"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text())
                    sessions.append(
                        {
                            "session_id": data.get("session_id"),
                            "session_name": data.get("session_name"),
                            "folder": str(folder),
                            "folder_name": folder.name,
                            "created_at": data.get("created_at"),
                        },
                    )
                except (json.JSONDecodeError, OSError):
                    continue

            if len(sessions) >= limit:
                break

        return sessions

    async def sync_session_to_db(
        self,
        session_id: str,
        folder_path: Path,
        db_session: "AsyncSession",
    ) -> None:
        """Update the database session record with the workspace path.

        Args:
            session_id: UUID of the session
            folder_path: Path to the session folder
            db_session: SQLAlchemy async session
        """
        from sqlalchemy import update

        from .db.schema import Session as DBSession

        stmt = (
            update(DBSession)
            .where(DBSession.id == uuid.UUID(session_id))
            .values(
                workspace_path=str(folder_path),
            )
        )
        await db_session.execute(stmt)
        await db_session.commit()
        logger.debug(f"[SESSION_MANAGER] Synced session {session_id} to DB with path {folder_path}")


# Singleton instance
session_manager = SessionPathManager()
