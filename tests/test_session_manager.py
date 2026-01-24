"""Tests for SessionPathManager

Tests session-centric architecture:
- Session folder creation
- Session lookup by ID and name
- Global folder isolation
- Project linking to sessions
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


class TestSessionPathManager:
    """Test SessionPathManager functionality."""

    @pytest.fixture
    def temp_config_root(self, tmp_path: Path):
        """Create a temporary CONFIG_ROOT for testing."""
        config_root = tmp_path / ".config" / "atlastrinity"
        config_root.mkdir(parents=True)
        return config_root

    @pytest.fixture
    def session_manager(self, temp_config_root: Path):
        """Create a SessionPathManager with temp directories."""
        # Patch CONFIG_ROOT before importing
        with patch("src.brain.session_manager.CONFIG_ROOT", temp_config_root):
            from src.brain.session_manager import SessionPathManager
            manager = SessionPathManager()
            # Override paths to use temp directory
            manager.vibe_workspace = temp_config_root / "vibe_workspace"
            manager.sessions_root = manager.vibe_workspace / "sessions"
            manager.global_folder = manager.vibe_workspace / "global"
            manager.workspace = temp_config_root / "workspace"
            manager._ensure_directories()
            return manager

    def test_directory_creation(self, session_manager):
        """Test that base directories are created."""
        assert session_manager.sessions_root.exists()
        assert session_manager.global_folder.exists()
        assert session_manager.workspace.exists()

    def test_create_session_folder(self, session_manager):
        """Test creating a new session folder."""
        session_id = str(uuid.uuid4())
        folder = session_manager.get_or_create_session_folder(session_id)

        assert folder.exists()
        assert folder.is_dir()
        assert folder.parent == session_manager.sessions_root

        # Check manifest file
        manifest = folder / ".session_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data["session_id"] == session_id

    def test_create_session_folder_with_name(self, session_manager):
        """Test creating a session folder with custom name."""
        session_id = str(uuid.uuid4())
        session_name = "Fix login bug"
        folder = session_manager.get_or_create_session_folder(session_id, session_name)

        manifest = folder / ".session_manifest.json"
        data = json.loads(manifest.read_text())
        assert data["session_name"] == session_name

    def test_get_existing_session_folder(self, session_manager):
        """Test getting an existing session folder."""
        session_id = str(uuid.uuid4())
        
        # Create folder
        folder1 = session_manager.get_or_create_session_folder(session_id)
        
        # Get same folder again
        folder2 = session_manager.get_or_create_session_folder(session_id)
        
        assert folder1 == folder2

    def test_get_session_folder_not_found(self, session_manager):
        """Test that non-existent session returns None."""
        fake_id = str(uuid.uuid4())
        result = session_manager.get_session_folder(fake_id)
        assert result is None

    def test_get_global_folder(self, session_manager):
        """Test getting the global folder."""
        global_folder = session_manager.get_global_folder()
        assert global_folder.exists()
        assert global_folder == session_manager.global_folder

    def test_find_session_by_path(self, session_manager):
        """Test finding session ID from a file path."""
        session_id = str(uuid.uuid4())
        folder = session_manager.get_or_create_session_folder(session_id)

        # Create a project file within the session
        project = folder / "my_project"
        project.mkdir()
        test_file = project / "main.py"
        test_file.write_text("# test")

        # Find session from file path
        found_id = session_manager.find_session_by_path(test_file)
        assert found_id == session_id

    def test_find_session_by_path_outside_sessions(self, session_manager):
        """Test that paths outside sessions return None."""
        result = session_manager.find_session_by_path("/tmp/some/file.py")
        assert result is None

    def test_find_session_by_name(self, session_manager):
        """Test finding session by project name."""
        session_id = str(uuid.uuid4())
        folder = session_manager.get_or_create_session_folder(session_id, "my_project_session")

        # Create a project folder
        project = folder / "calculator_app"
        project.mkdir()

        # Find by project folder name
        found_id = session_manager.find_session_by_name("calculator")
        assert found_id == session_id

    def test_find_session_by_session_name(self, session_manager):
        """Test finding session by session name."""
        session_id = str(uuid.uuid4())
        session_manager.get_or_create_session_folder(session_id, "Fix authentication bug")

        found_id = session_manager.find_session_by_name("authentication")
        assert found_id == session_id

    def test_is_global_path(self, session_manager):
        """Test checking if path is in global folder."""
        global_file = session_manager.global_folder / "utils.py"
        session_file = session_manager.sessions_root / "2026-01-01_abc123" / "main.py"

        assert session_manager.is_global_path(global_file) is True
        assert session_manager.is_global_path(session_file) is False
        assert session_manager.is_global_path("/tmp/other.py") is False

    def test_is_session_path(self, session_manager):
        """Test checking if path is in a session folder."""
        session_id = str(uuid.uuid4())
        folder = session_manager.get_or_create_session_folder(session_id)
        
        session_file = folder / "project" / "main.py"
        global_file = session_manager.global_folder / "utils.py"

        # Need to create the path structure for find_session_by_path to work
        (folder / "project").mkdir()
        session_file.write_text("# test")

        assert session_manager.is_session_path(session_file) is True
        assert session_manager.is_session_path(global_file) is False

    def test_list_sessions(self, session_manager):
        """Test listing sessions."""
        # Create multiple sessions
        ids = [str(uuid.uuid4()) for _ in range(3)]
        for i, session_id in enumerate(ids):
            session_manager.get_or_create_session_folder(session_id, f"Session {i}")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 3
        
        # Check that all sessions have required fields
        for session in sessions:
            assert "session_id" in session
            assert "folder" in session
            assert "created_at" in session

    def test_link_project_to_session(self, session_manager, tmp_path: Path):
        """Test linking an existing project to a session."""
        # Create an external project
        external_project = tmp_path / "external_project"
        external_project.mkdir()
        (external_project / "main.py").write_text("# external")

        session_id = str(uuid.uuid4())
        session_manager.get_or_create_session_folder(session_id)

        # Link project
        link = session_manager.link_project_to_session(external_project, session_id)
        
        assert link is not None
        assert link.is_symlink()
        assert link.resolve() == external_project.resolve()

    def test_link_nonexistent_project(self, session_manager):
        """Test linking a non-existent project returns None."""
        session_id = str(uuid.uuid4())
        session_manager.get_or_create_session_folder(session_id)

        result = session_manager.link_project_to_session("/fake/path", session_id)
        assert result is None

    def test_folder_naming_pattern(self, session_manager):
        """Test that folders follow the naming pattern: date_shortid."""
        session_id = str(uuid.uuid4())
        folder = session_manager.get_or_create_session_folder(session_id)

        # Check naming pattern: YYYY-MM-DD_shortid
        name = folder.name
        parts = name.split("_")
        assert len(parts) >= 2

        date_part = parts[0]
        assert len(date_part) == 10  # YYYY-MM-DD
        assert date_part.count("-") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
