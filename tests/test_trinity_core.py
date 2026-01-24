#!/usr/bin/env python3
"""Comprehensive Trinity Core Tests
Tests Atlas, Orchestrator, and core Trinity functionality
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.agents.atlas import Atlas, TaskPlan
from src.brain.context import shared_context


class TestAtlasCore:
    """Test core Atlas functionality"""

    @pytest.mark.asyncio
    async def test_atlas_initialization(self):
        """Test Atlas initializes correctly"""
        atlas = Atlas(model_name="raptor-mini")
        assert atlas.NAME == "ATLAS"
        assert atlas.VOICE == "Dmytro"
        assert atlas.llm is not None
        assert atlas.current_plan is None
        assert isinstance(atlas.history, list)

    @pytest.mark.asyncio
    async def test_atlas_chat_detection_greetings(self):
        """Test Atlas correctly detects greetings"""
        atlas = Atlas()

        test_cases = [
            "привіт",
            "здоров",
            "як справи",
            "як ти",
            "дякую",
            "hi",
        ]

        for greeting in test_cases:
            result = await atlas.analyze_request(greeting)
            assert result["intent"] == "chat", f"Failed for: {greeting}"
            assert "initial_response" in result

    @pytest.mark.asyncio
    async def test_atlas_task_detection(self):
        """Test Atlas correctly detects tasks"""
        atlas = Atlas()

        # Mock LLM response
        class MockLLM:
            async def ainvoke(self, *args, **kwargs):
                class MockResponse:
                    content = '{"intent": "task", "reason": "Це завдання", "enriched_request": "Open terminal", "complexity": "low", "initial_response": null}'

                return MockResponse()

        atlas.llm = MockLLM()  # type: ignore

        result = await atlas.analyze_request("відкрий термінал")
        assert result["intent"] == "task"
        assert result["initial_response"] is None

    @pytest.mark.asyncio
    async def test_atlas_error_handling(self):
        """Test Atlas handles LLM errors gracefully"""
        atlas = Atlas()

        # Mock failing LLM
        class FailingLLM:
            async def ainvoke(self, *args, **kwargs):
                raise Exception("API Error")

        atlas.llm = FailingLLM()  # type: ignore

        # Should fallback to heuristic detection
        result1 = await atlas.analyze_request("привіт")
        assert result1["intent"] == "chat"

        result2 = await atlas.analyze_request("відкрий terminal")
        assert result2["intent"] == "task"

    def test_atlas_voice_messages(self):
        """Test Atlas generates correct voice messages"""
        atlas = Atlas()

        msg1 = atlas.get_voice_message("plan_created", steps=5)
        assert "5 пунктів" in msg1

        msg2 = atlas.get_voice_message("delegating")
        assert "Тетяно" in msg2

    def test_atlas_response_parsing(self):
        """Test JSON response parsing"""
        atlas = Atlas()

        # Valid JSON
        json_str = '{"key": "value", "number": 42}'
        result = atlas._parse_response(json_str)
        assert result["key"] == "value"
        assert result["number"] == 42

        # JSON embedded in text
        text_with_json = 'Some text before {"embedded": true} and after'
        result2 = atlas._parse_response(text_with_json)
        assert result2["embedded"] is True

        # Invalid JSON
        result3 = atlas._parse_response("Not JSON at all")
        assert "raw" in result3


class TestSharedContext:
    """Test shared context functionality"""

    def test_context_path_resolution(self):
        """Test path resolution"""
        # Test tilde expansion
        path1 = shared_context.resolve_path("~/test.txt")
        assert path1.startswith("/Users/")
        assert "test.txt" in path1

    def test_context_updates(self):
        """Test context tracking"""
        test_path = "/Users/test/file.txt"
        shared_context.update_path(test_path, "write")

        context_dict = shared_context.to_dict()
        assert test_path in context_dict["recent_files"]


class TestTaskPlan:
    """Test TaskPlan dataclass"""

    def test_plan_creation(self):
        """Test plan creation"""
        plan = TaskPlan(
            id="test_123",
            goal="Test goal",
            steps=[{"id": 1, "action": "Test action", "tool": "terminal"}],
        )

        assert plan.id == "test_123"
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1
        assert plan.status == "pending"
        assert plan.created_at is not None


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_atlas_parse_response_speed(self):
        """Test JSON parsing is fast"""
        atlas = Atlas()

        import time

        large_json = '{"data": ' + str(list(range(1000))) + "}"

        start = time.time()
        for _ in range(100):
            atlas._parse_response(large_json)
        elapsed = time.time() - start

        # Should parse 100 times in less than 0.1 seconds
        assert elapsed < 0.1, f"Parsing too slow: {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
