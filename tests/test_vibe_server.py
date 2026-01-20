"""
Unit tests for vibe_server.py (v3.0 Hyper-Refactored)

Tests:
1. Configuration loading (vibe_config.py)
2. Tool pattern matching (glob/regex)
3. CLI argument building
4. Prompt file handling
5. Instruction cleanup
"""

import os
import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Need to patch before importing
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVibeConfig:
    """Tests for vibe_config.py configuration system."""

    def test_default_config_loads(self):
        """Default configuration should load without errors."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig()
        assert config.active_model == "devstral-2"
        assert config.max_turns == 10
        assert config.timeout_s == 600.0

    def test_tool_pattern_glob(self):
        """Glob patterns should match tool names correctly."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig(
            enabled_tools=["serena_*", "read_file"],
            disabled_tools=["mcp_*"]
        )
        
        # Enabled by glob
        assert config.is_tool_enabled("serena_list")
        assert config.is_tool_enabled("serena_query")
        
        # Enabled by exact match
        assert config.is_tool_enabled("read_file")
        
        # Disabled by glob (takes precedence)
        assert not config.is_tool_enabled("mcp_fetch")
        assert not config.is_tool_enabled("mcp_server_tool")
        
        # Not in enabled list
        assert not config.is_tool_enabled("bash")

    def test_tool_pattern_regex(self):
        """Regex patterns (re: prefix) should match correctly."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig(
            enabled_tools=["re:^db_.*$"],
            disabled_tools=["re:^dangerous_.*"]
        )
        
        # Enabled by regex
        assert config.is_tool_enabled("db_query")
        assert config.is_tool_enabled("db_insert")
        
        # Disabled by regex
        assert not config.is_tool_enabled("dangerous_delete")
        
        # Not matching
        assert not config.is_tool_enabled("other_tool")

    def test_tool_pattern_empty_enabled_means_all(self):
        """Empty enabled_tools means all tools are enabled by default."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig(
            enabled_tools=[],
            disabled_tools=["bash"]
        )
        
        # All enabled except disabled
        assert config.is_tool_enabled("read_file")
        assert config.is_tool_enabled("write_file")
        assert config.is_tool_enabled("any_random_tool")
        
        # But bash is disabled
        assert not config.is_tool_enabled("bash")

    def test_cli_args_building(self):
        """CLI arguments should be built correctly from config."""
        from src.mcp_server.vibe_config import AgentMode, VibeConfig
        
        config = VibeConfig(
            active_model="devstral-2",
            default_mode=AgentMode.AUTO_APPROVE,
            max_turns=10
        )
        
        args = config.to_cli_args(
            prompt="Test prompt",
            mode=AgentMode.AUTO_APPROVE,
        )
        
        assert "-p" in args
        assert "Test prompt" in args
        assert "--output" in args
        assert "streaming" in args
        assert "--auto-approve" in args

    def test_cli_args_model_override(self):
        """Model override should be included in CLI args."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig()
        
        args = config.to_cli_args(
            prompt="Test",
            model="gpt-4o",
        )
        
        assert "--model" in args
        idx = args.index("--model")
        assert args[idx + 1] == "gpt-4o"

    def test_cli_args_session_resume(self):
        """Session ID should be included for resume."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig()
        
        args = config.to_cli_args(
            prompt="Continue",
            session_id="abc123",
        )
        
        assert "--session" in args
        idx = args.index("--session")
        assert args[idx + 1] == "abc123"

    def test_provider_api_key_check(self):
        """Provider availability should check for API key."""
        from src.mcp_server.vibe_config import ApiStyle, Backend, ProviderConfig
        
        provider = ProviderConfig(
            name="test",
            api_base="https://test.ai/v1",
            api_key_env_var="TEST_API_KEY",
            api_style=ApiStyle.OPENAI,
            backend=Backend.GENERIC
        )
        
        # Without env var set, should be unavailable
        assert not provider.is_available()
        
        # With env var set
        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            assert provider.is_available()

    def test_model_lookup_by_alias(self):
        """Model lookup by alias should work."""
        from src.mcp_server.vibe_config import ModelConfig, VibeConfig
        
        config = VibeConfig(
            models=[
                ModelConfig(
                    name="gpt-4o",
                    provider="openrouter",
                    alias="gpt4",
                    temperature=0.3
                )
            ]
        )
        
        model = config.get_model_by_alias("gpt4")
        assert model is not None
        assert model.name == "gpt-4o"
        
        # Non-existent
        assert config.get_model_by_alias("nonexistent") is None


class TestPreparePromptArg:
    """Tests for handle_long_prompt function."""

    @pytest.fixture
    def mock_instructions_dir(self, tmp_path):
        """Create a temporary instructions directory."""
        instructions_dir = tmp_path / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        return str(instructions_dir)

    def test_small_prompt_no_file_created(self, mock_instructions_dir):
        """Small prompts (<= 2000 chars) should not create files."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import handle_long_prompt

            small_prompt = "A" * 1999
            result, file_path = handle_long_prompt(small_prompt)

            assert result == small_prompt
            assert file_path is None
            # No files should be created
            assert len(list(Path(mock_instructions_dir).glob("*.md"))) == 0

    def test_large_prompt_creates_file_in_global_dir(self, mock_instructions_dir):
        """Large prompts should create files in INSTRUCTIONS_DIR."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import handle_long_prompt

            large_prompt = "B" * 2500

            # Pass a different cwd - should be ignored
            result, file_path = handle_long_prompt(large_prompt, cwd="/some/random/path")

            # File should be in global instructions dir
            assert file_path is not None
            assert mock_instructions_dir in file_path
            assert "/some/random/path" not in file_path
            assert os.path.exists(file_path)

            # Check content
            with open(file_path, "r") as f:
                content = f.read()
                assert "# VIBE INSTRUCTIONS" in content
                assert large_prompt in content

    def test_prompt_file_contains_full_path(self, mock_instructions_dir):
        """The returned prompt arg should contain the full path."""
        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", mock_instructions_dir):
            from src.mcp_server.vibe_server import handle_long_prompt

            large_prompt = "C" * 3000
            result, file_path = handle_long_prompt(large_prompt)

            # Result should reference the file path
            assert file_path in result or mock_instructions_dir in result


class TestCleanupOldInstructions:
    """Tests for cleanup_old_instructions function."""

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
            from src.mcp_server.vibe_server import cleanup_old_instructions

            cleaned = cleanup_old_instructions(max_age_hours=24)

            # Old file should be removed
            assert cleaned == 1
            assert not os.path.exists(old_file)
            # New file should remain
            assert os.path.exists(new_file)

    def test_cleanup_nonexistent_dir(self, tmp_path):
        """Should handle nonexistent directory gracefully."""
        nonexistent = str(tmp_path / "nonexistent")

        with patch("src.mcp_server.vibe_server.INSTRUCTIONS_DIR", nonexistent):
            from src.mcp_server.vibe_server import cleanup_old_instructions

            cleaned = cleanup_old_instructions(max_age_hours=24)
            assert cleaned == 0


class TestAgentMode:
    """Tests for AgentMode enum and mode switching."""

    def test_agent_modes_exist(self):
        """All expected agent modes should exist."""
        from src.mcp_server.vibe_config import AgentMode
        
        assert AgentMode.DEFAULT.value == "default"
        assert AgentMode.PLAN.value == "plan"
        assert AgentMode.ACCEPT_EDITS.value == "accept-edits"
        assert AgentMode.AUTO_APPROVE.value == "auto-approve"

    def test_mode_from_string(self):
        """Mode should be creatable from string."""
        from src.mcp_server.vibe_config import AgentMode
        
        assert AgentMode("auto-approve") == AgentMode.AUTO_APPROVE
        assert AgentMode("plan") == AgentMode.PLAN


class TestEnvironmentConfig:
    """Tests for environment variable handling."""

    def test_get_environment_basic(self):
        """Environment should include basic CLI settings."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig()
        env = config.get_environment()
        
        assert env["TERM"] == "dumb"
        assert env["NO_COLOR"] == "1"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_get_environment_custom_vibe_home(self):
        """Custom VIBE_HOME should be included in environment."""
        from src.mcp_server.vibe_config import VibeConfig
        
        config = VibeConfig(vibe_home="/custom/vibe/home")
        env = config.get_environment()
        
        assert env["VIBE_HOME"] == "/custom/vibe/home"


class TestToolPermissions:
    """Tests for tool permission configuration."""

    def test_tool_permission_levels(self):
        """All permission levels should be available."""
        from src.mcp_server.vibe_config import ToolPermission
        
        assert ToolPermission.ALWAYS.value == "always"
        assert ToolPermission.ASK.value == "ask"
        assert ToolPermission.NEVER.value == "never"

    def test_get_tool_permission(self):
        """Tool permissions should be retrievable from config."""
        from src.mcp_server.vibe_config import ToolConfig, ToolPermission, VibeConfig
        
        config = VibeConfig(
            tools={
                "bash": ToolConfig(permission=ToolPermission.NEVER),
                "read_file": ToolConfig(permission=ToolPermission.ALWAYS),
            }
        )
        
        assert config.get_tool_permission("bash") == ToolPermission.NEVER
        assert config.get_tool_permission("read_file") == ToolPermission.ALWAYS
        # Default for unknown tools
        assert config.get_tool_permission("unknown") == ToolPermission.ASK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
