"""
Unit tests for vibe_server.py

Tests:
1. _prepare_prompt_arg stores files in global INSTRUCTIONS_DIR
2. _cleanup_old_instructions removes old files
3. Small prompts don't create files
"""

import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Need to patch before importing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreparePromptArg:
    """Tests for _prepare_prompt_arg function."""

    @pytest.fixture
    def mock_instructions_dir(self, tmp_path):
        """Create a temporary instructions directory."""
        instructions_dir = tmp_path / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        return str(instructions_dir)

    def test_small_prompt_no_file_created(self, mock_instructions_dir):
        """Small prompts (<= 2000 chars) should not create files."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import _prepare_prompt_arg
            
            small_prompt = "A" * 1999
            result, file_path = _prepare_prompt_arg(small_prompt)
            
            assert result == small_prompt
            assert file_path is None
            # No files should be created
            assert len(list(Path(mock_instructions_dir).glob("*.md"))) == 0

    def test_large_prompt_creates_file_in_global_dir(self, mock_instructions_dir):
        """Large prompts should create files in INSTRUCTIONS_DIR, ignoring cwd."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import _prepare_prompt_arg
            
            large_prompt = "B" * 2500
            
            # Pass a different cwd - should be ignored
            result, file_path = _prepare_prompt_arg(large_prompt, cwd="/some/random/path")
            
            # File should be in global instructions dir
            assert file_path is not None
            assert mock_instructions_dir in file_path
            assert "/some/random/path" not in file_path
            assert os.path.exists(file_path)
            
            # Check content
            with open(file_path, "r") as f:
                content = f.read()
                assert "# INSTRUCTIONS FOR VIBE AGENT" in content
                assert large_prompt in content

    def test_prompt_file_contains_full_path(self, mock_instructions_dir):
        """The returned prompt arg should contain the full path for Vibe to find."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import _prepare_prompt_arg
            
            large_prompt = "C" * 3000
            result, file_path = _prepare_prompt_arg(large_prompt)
            
            # Result should contain full path
            assert file_path in result or mock_instructions_dir in result


class TestCleanupOldInstructions:
    """Tests for _cleanup_old_instructions function."""

    @pytest.fixture
    def mock_instructions_dir_with_files(self, tmp_path):
        """Create directory with old and new instruction files."""
        instructions_dir = tmp_path / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create old file (modified 30 hours ago)
        old_file = instructions_dir / "vibe_instructions_1000000000_abc123.md"
        old_file.write_text("old content")
        old_time = (datetime.now() - timedelta(hours=30)).timestamp()
        os.utime(old_file, (old_time, old_time))
        
        # Create new file (modified just now)
        new_file = instructions_dir / "vibe_instructions_9999999999_def456.md"
        new_file.write_text("new content")
        
        return str(instructions_dir), str(old_file), str(new_file)

    def test_cleanup_removes_old_files(self, mock_instructions_dir_with_files):
        """Should remove files older than max_age_hours."""
        instructions_dir, old_file, new_file = mock_instructions_dir_with_files
        
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", instructions_dir):
            from src.mcp_server.vibe_server import _cleanup_old_instructions
            
            cleaned = _cleanup_old_instructions(max_age_hours=24)
            
            # Old file should be removed
            assert cleaned == 1
            assert not os.path.exists(old_file)
            # New file should remain
            assert os.path.exists(new_file)

    def test_cleanup_nonexistent_dir(self, tmp_path):
        """Should handle nonexistent directory gracefully."""
        nonexistent = str(tmp_path / "nonexistent")
        
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", nonexistent):
            from src.mcp_server.vibe_server import _cleanup_old_instructions
            
            cleaned = _cleanup_old_instructions(max_age_hours=24)
            assert cleaned == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
