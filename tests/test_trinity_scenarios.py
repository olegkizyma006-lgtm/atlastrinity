import asyncio
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure Copilot provider doesn't require real credentials in import-time code paths
os.environ.setdefault("COPILOT_API_KEY", "dummy")


# --- Minimal dependency stubs (to keep tests runnable without heavy optional deps) ---

# Stub providers.copilot (agents import this)
providers_mod = types.ModuleType("providers")
providers_copilot_mod = types.ModuleType("providers.copilot")


class _StubCopilotLLM:
    def __init__(self, *args, **kwargs):
        pass

    async def ainvoke(self, *args, **kwargs):
        return SimpleNamespace(content="{}")


providers_copilot_mod.CopilotLLM = _StubCopilotLLM  # type: ignore
providers_mod.copilot = providers_copilot_mod  # type: ignore
sys.modules.setdefault("providers", providers_mod)
sys.modules.setdefault("providers.copilot", providers_copilot_mod)


# Stub langgraph.graph used during Trinity graph build
langgraph_mod = types.ModuleType("langgraph")
langgraph_graph_mod = types.ModuleType("langgraph.graph")


class _StubStateGraph:
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, *args, **kwargs):
        return None

    def set_entry_point(self, *args, **kwargs):
        return None

    def add_edge(self, *args, **kwargs):
        return None

    def add_conditional_edges(self, *args, **kwargs):
        return None

    def compile(self):
        return SimpleNamespace()


langgraph_graph_mod.StateGraph = _StubStateGraph  # type: ignore
langgraph_graph_mod.END = "__END__"  # type: ignore
langgraph_graph_mod.add_messages = lambda *args, **kwargs: None  # type: ignore
langgraph_mod.graph = langgraph_graph_mod  # type: ignore
sys.modules.setdefault("langgraph", langgraph_mod)
sys.modules.setdefault("langgraph.graph", langgraph_graph_mod)


# Stub langchain_core.messages (orchestrator imports it at module import time)
langchain_core_mod = types.ModuleType("langchain_core")
langchain_core_messages_mod = types.ModuleType("langchain_core.messages")


class _BaseMessage:
    def __init__(self, content=None, **kwargs):
        self.content = content


class _HumanMessage(_BaseMessage):
    pass


class _AIMessage(_BaseMessage):
    pass


class _SystemMessage(_BaseMessage):
    pass


langchain_core_messages_mod.BaseMessage = _BaseMessage  # type: ignore
langchain_core_messages_mod.HumanMessage = _HumanMessage  # type: ignore
langchain_core_messages_mod.AIMessage = _AIMessage  # type: ignore
langchain_core_messages_mod.SystemMessage = _SystemMessage  # type: ignore
langchain_core_mod.messages = langchain_core_messages_mod  # type: ignore

sys.modules.setdefault("langchain_core", langchain_core_mod)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages_mod)


# Stub ukrainian_tts to avoid optional import errors during TTS calls
sys.modules.setdefault("ukrainian_tts", types.ModuleType("ukrainian_tts"))
sys.modules.setdefault("ukrainian_tts.tts", types.ModuleType("ukrainian_tts.tts"))


# Stub src.brain.mcp_manager to avoid importing MCP client stack in unit tests
brain_mcp_manager_mod = types.ModuleType("src.brain.mcp_manager")


class _StubMCPManager:
    def start_health_monitoring(self, interval: int = 60):
        return None


brain_mcp_manager_mod.mcp_manager = _StubMCPManager()  # type: ignore
sys.modules.setdefault("src.brain.mcp_manager", brain_mcp_manager_mod)


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


from src.brain.agents.tetyana import StepResult
from src.brain.orchestrator import Trinity


class DummyVoice:
    async def speak(self, agent_id: str, text: str):
        return None


class StubNotifications:
    def __init__(self):
        self.progress_calls = []
        self.stuck_alerts = []
        self.approvals = []
        self.completions = []
        self.next_approval = True

    def show_progress(self, current_step, total_steps, description=""):
        self.progress_calls.append((current_step, total_steps, description))
        return True

    def send_stuck_alert(self, step_id, error, attempts=0):
        self.stuck_alerts.append((step_id, error, attempts))
        return True

    def request_approval(self, action, risk_level="medium"):
        self.approvals.append((action, risk_level))
        return self.next_approval

    def show_completion(self, task, success, duration_seconds=0):
        self.completions.append((task, success, duration_seconds))
        return True


def _plan(steps):
    return SimpleNamespace(steps=steps, goal="Тестовий план")


@pytest.fixture
def fast_sleep(monkeypatch):
    import src.brain.orchestrator as orch

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(orch.asyncio, "sleep", _no_sleep)


@pytest.fixture
def isolated_globals(monkeypatch):
    import src.brain.orchestrator as orch

    # Avoid Redis / memory / consolidation side-effects during unit tests
    monkeypatch.setattr(orch.state_manager, "available", False, raising=False)
    monkeypatch.setattr(orch.long_term_memory, "available", False, raising=False)  # type: ignore
    monkeypatch.setattr(
        orch.consolidation_module,
        "should_consolidate",
        lambda: False,
        raising=False,
    )


@pytest.fixture
def stub_notifications(monkeypatch):
    import src.brain.orchestrator as orch

    stub = StubNotifications()
    monkeypatch.setattr(orch, "notifications", stub, raising=True)
    return stub


@pytest.fixture
def trinity_base(isolated_globals, stub_notifications, fast_sleep):
    t = Trinity()
    t.voice = DummyVoice()  # type: ignore
    return t


def test_chat_intent_returns_chat_result(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "chat", "initial_response": "Привіт!"}

        async def chat(self, user_request: str, history=None):
            return "Привіт!"

    trinity_base.atlas = MockAtlas()

    res = asyncio.run(trinity_base.run("привіт"))
    assert res["status"] == "completed"
    assert res["type"] == "chat"
    assert "Привіт" in res["result"]


def test_planning_error_returns_error(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            raise RuntimeError("boom")

    trinity_base.atlas = MockAtlas()

    res = asyncio.run(trinity_base.run("зроби щось"))
    assert res["status"] == "error"
    assert "boom" in res["error"]


def test_empty_plan_returns_no_steps_message(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            return _plan([])

        def get_voice_message(self, *args, **kwargs):
            return ""

        async def help_tetyana(self, step_id, error):
            return {"voice_message": "", "alternative_steps": []}

    trinity_base.atlas = MockAtlas()

    res = asyncio.run(trinity_base.run("зроби план"))
    assert res["status"] == "completed"
    assert res["type"] == "chat"
    assert "Не знайдено кроків" in res["result"]


def test_simple_execution_appends_step_result(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            return _plan(
                [
                    {
                        "id": 1,
                        "action": "Echo",
                        "tool": "terminal",
                        "requires_verification": False,
                    },
                ],
            )

        def get_voice_message(self, *args, **kwargs):
            return ""

    class MockTetyana:
        async def execute_step(self, step, attempt: int = 1):
            return StepResult(step_id=step.get("id"), success=True, result="OK", error=None)

        def get_voice_message(self, *args, **kwargs):
            return ""

    trinity_base.atlas = MockAtlas()
    trinity_base.tetyana = MockTetyana()

    res = asyncio.run(trinity_base.run("виконай"))
    assert res["status"] == "completed"
    assert isinstance(res["result"], list)
    assert len(res["result"]) == 1
    assert res["result"][0]["success"] is True


def test_verification_rejection_marks_step_failed(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            return _plan(
                [
                    {
                        "id": 1,
                        "action": "Do",
                        "tool": "terminal",
                        "requires_verification": True,
                        "expected_result": "ok",
                    },
                ],
            )

        def get_voice_message(self, *args, **kwargs):
            return ""

        async def help_tetyana(self, step_id, error):
            return {"voice_message": "", "alternative_steps": []}

    class MockTetyana:
        async def execute_step(self, step, attempt: int = 1):
            return StepResult(step_id=step.get("id"), success=True, result="OK", error=None)

        def get_voice_message(self, *args, **kwargs):
            return ""

    class MockGrisha:
        async def take_screenshot(self):
            return ""

        async def verify_step(self, step, result, screenshot_path=None):
            return SimpleNamespace(
                verified=False,
                description="not ok",
                issues=["mismatch"],
                voice_message="Ні",
            )

    trinity_base.atlas = MockAtlas()
    trinity_base.tetyana = MockTetyana()
    trinity_base.grisha = MockGrisha()

    res = asyncio.run(trinity_base.run("виконай"))
    assert res["status"] == "failed"
    step_results = res["result"]
    assert isinstance(step_results, list)
    assert any("Grisha rejected" in (r.get("error") or "") for r in step_results)


def test_verification_crash_is_caught_and_logged(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            return _plan(
                [
                    {
                        "id": 1,
                        "action": "Do",
                        "tool": "terminal",
                        "requires_verification": True,
                        "expected_result": "ok",
                    },
                ],
            )

        def get_voice_message(self, *args, **kwargs):
            return ""

        async def help_tetyana(self, step_id, error):
            return {"voice_message": "", "alternative_steps": []}

    class MockTetyana:
        async def execute_step(self, step, attempt: int = 1):
            return StepResult(step_id=step.get("id"), success=True, result="OK", error=None)

        def get_voice_message(self, *args, **kwargs):
            return ""

    class MockGrisha:
        async def take_screenshot(self):
            return ""

        async def verify_step(self, step, result, screenshot_path=None):
            raise RuntimeError("Verifier down")

    trinity_base.atlas = MockAtlas()
    trinity_base.tetyana = MockTetyana()
    trinity_base.grisha = MockGrisha()

    res = asyncio.run(trinity_base.run("виконай"))
    assert res["status"] == "failed"
    step_results = res["result"]
    assert isinstance(step_results, list)
    assert any("Verification system error" in (r.get("error") or "") for r in step_results)

    logs = trinity_base.state.get("logs", [])
    assert any("Verification crashed" in str(l.get("message")) for l in logs)


def test_retries_then_user_rejects_recovery_aborts(trinity_base, stub_notifications):
    stub_notifications.next_approval = False

    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            return {"intent": "task", "reason": "Ок"}

        async def create_plan(self, analysis):
            return _plan(
                [
                    {
                        "id": 1,
                        "action": "Do",
                        "tool": "terminal",
                        "requires_verification": False,
                    },
                ],
            )

        async def help_tetyana(self, step_id, error):
            return {"voice_message": "", "alternative_steps": []}

        def get_voice_message(self, *args, **kwargs):
            return ""

    class MockTetyana:
        async def execute_step(self, step, attempt: int = 1):
            return StepResult(
                step_id=step.get("id"),
                success=False,
                result="NO",
                error=f"fail-{attempt}",
            )

        def get_voice_message(self, *args, **kwargs):
            return ""

    trinity_base.atlas = MockAtlas()
    trinity_base.tetyana = MockTetyana()

    res = asyncio.run(trinity_base.run("виконай"))
    assert res["status"] == "failed"
    assert "Task aborted" in res["error"]
    assert len(stub_notifications.stuck_alerts) == 1
    assert len(stub_notifications.approvals) == 1


def test_subtask_step_triggers_recursive_run(trinity_base):
    class MockAtlas:
        async def analyze_request(self, user_request: str, history=None, context=None):
            if "привіт" in user_request.lower():
                return {"intent": "chat", "initial_response": "Привіт!"}
            return {"intent": "task", "reason": "Ок"}

        async def chat(self, user_request: str, history=None):
            return "Привіт!"

        async def create_plan(self, analysis):
            return _plan(
                [
                    {
                        "id": 1,
                        "action": "привіт",
                        "type": "subtask",
                        "tool": "subtask",
                        "requires_verification": False,
                    },
                ],
            )

        def get_voice_message(self, *args, **kwargs):
            return ""

    class MockTetyana:
        async def execute_step(self, step, attempt: int = 1):
            return StepResult(step_id=step.get("id"), success=True, result="OK", error=None)

        def get_voice_message(self, *args, **kwargs):
            return ""

    trinity_base.atlas = MockAtlas()
    trinity_base.tetyana = MockTetyana()

    res = asyncio.run(trinity_base.run("зроби підзадачу"))
    assert res["status"] == "completed"
    assert isinstance(res["result"], list)
    assert len(res["result"]) == 1
    assert res["result"][0]["success"] is True
