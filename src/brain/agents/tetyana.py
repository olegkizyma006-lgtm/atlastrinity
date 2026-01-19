"""
Tetyana - The Executor

Role: macOS interaction, executing atomic plan steps
Voice: Tetiana (female)
Model: GPT-4.1
"""

import asyncio
import json
import os

# Robust path handling for both Dev and Production (Packaged)
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..mcp_manager import mcp_manager

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dev = os.path.join(current_dir, "..", "..", "..")
root_prod = os.path.join(current_dir, "..", "..")

for r in [root_dev, root_prod]:
    abs_r = os.path.abspath(r)
    if abs_r not in sys.path:
        sys.path.insert(0, abs_r)

from providers.copilot import CopilotLLM  # noqa: E402

from ..config_loader import config  # noqa: E402
from ..context import shared_context  # noqa: E402
from ..prompts import AgentPrompts  # noqa: E402
from .base_agent import BaseAgent  # noqa: E402


@dataclass
class StepResult:
    """Result of step execution"""

    step_id: str
    success: bool
    result: str
    screenshot_path: Optional[str] = None
    voice_message: Optional[str] = None
    error: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    thought: Optional[str] = None
    is_deviation: bool = False
    deviation_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """Convert StepResult to dictionary"""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "result": self.result,
            "screenshot_path": self.screenshot_path,
            "voice_message": self.voice_message,
            "error": self.error,
            "tool_call": self.tool_call,
            "thought": self.thought,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class Tetyana(BaseAgent):
    """
    Tetyana - The Executor

    Functions:
    - Executing atomic plan steps
    - Interacting with macOS (GUI/Terminal/Apps)
    - Progress reporting
    - Asking Atlas for help when stuck
    """

    # Tool schemas are now loaded from centralized mcp_registry
    # This eliminates duplication and ensures single source of truth
    _cached_schemas = None

    @classmethod
    def get_tool_schemas(cls) -> dict:
        """
        Get tool schemas from centralized registry.
        Cached after first access for performance.
        """
        if cls._cached_schemas is None:
            from ..mcp_registry import TOOL_SCHEMAS
            cls._cached_schemas = TOOL_SCHEMAS
        return cls._cached_schemas

    # Backwards compatibility property
    @property
    def MACOS_USE_SCHEMAS(self) -> dict:
        """Legacy property for backwards compatibility. Use get_tool_schemas() instead."""
        return self.get_tool_schemas()

    NAME = AgentPrompts.TETYANA["NAME"]
    DISPLAY_NAME = AgentPrompts.TETYANA["DISPLAY_NAME"]
    VOICE = AgentPrompts.TETYANA["VOICE"]
    COLOR = AgentPrompts.TETYANA["COLOR"]
    SYSTEM_PROMPT = AgentPrompts.TETYANA["SYSTEM_PROMPT"]


    def __init__(self, model_name: str = "grok-code-fast-1"):
        # Get model config (config.yaml > parameter > env variables)
        agent_config = config.get_agent_config("tetyana")
        final_model = model_name
        if model_name == "grok-code-fast-1":  # default parameter
            final_model = agent_config.get("model") or os.getenv("COPILOT_MODEL", "gpt-4.1")

        self.llm = CopilotLLM(model_name=final_model)

        # Specialized models for Reasoning and Reflexion
        reasoning_model = agent_config.get("reasoning_model") or os.getenv(
            "REASONING_MODEL", "raptor-mini"
        )
        reflexion_model = agent_config.get("reflexion_model") or os.getenv(
            "REFLEXION_MODEL", "gpt-5-mini"
        )

        self.reasoning_llm = CopilotLLM(model_name=reasoning_model)
        self.reflexion_llm = CopilotLLM(model_name=reflexion_model)

        # NEW: Vision model for complex GUI tasks (screenshot analysis)
        vision_model = agent_config.get("vision_model") or os.getenv("VISION_MODEL", "gpt-4o")
        self.vision_llm = CopilotLLM(model_name=vision_model, vision_model_name=vision_model)

        self.temperature = agent_config.get("temperature", 0.5)
        self.current_step: int = 0
        self.results: List[StepResult] = []
        self.attempt_count: int = 0
        
        # Track current PID for Vision analysis
        self._current_pid: Optional[int] = None

    async def get_grisha_feedback(self, step_id: int) -> Optional[str]:
        """Retrieve Grisha's detailed rejection report from notes or memory"""
        from ..logger import logger  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        # Try macos-use notes first (faster)
        try:
            result = await mcp_manager.dispatch_tool(
                "notes_get",
                {"name": f"Grisha Rejection Step {step_id}"}
            )

            # Normalize notes search result to a plain dict when possible
            notes_result = None
            try:
                if isinstance(result, dict):
                    notes_result = result
                elif hasattr(result, "structuredContent") and isinstance(
                    getattr(result, "structuredContent"), dict
                ):
                    notes_result = result.structuredContent.get("result", {})
                elif (
                    hasattr(result, "content")
                    and len(result.content) > 0
                    and hasattr(result.content[0], "text")
                ):
                    import json as _json  # noqa: E402

                    try:
                        notes_result = _json.loads(result.content[0].text)
                    except Exception:
                        notes_result = None
            except Exception:
                notes_result = None

            if (
                isinstance(notes_result, dict)
                and notes_result.get("success")
                and notes_result.get("notes")
            ):
                notes = notes_result.get("notes") or [notes_result] if isinstance(notes_result, dict) else []
                if notes and len(notes) > 0:
                    note_content = notes_result.get("content", "") or notes_result.get("body", "")

                    # Normalize read_note result
                    note_content = None
                    if isinstance(note_result, dict) and note_result.get("success"):
                        note_content = note_result.get("content", "")
                    elif hasattr(note_result, "structuredContent") and isinstance(
                        getattr(note_result, "structuredContent"), dict
                    ):
                        note_content = note_result.structuredContent.get("result", {}).get(
                            "content", ""
                        )
                    elif (
                        hasattr(note_result, "content")
                        and len(note_result.content) > 0
                        and hasattr(note_result.content[0], "text")
                    ):
                        try:
                            note_parsed = json.loads(note_result.content[0].text)
                            note_content = note_parsed.get("content", "")
                        except Exception:
                            note_content = None

                    if note_content:
                        logger.info(
                            f"[TETYANA] Retrieved Grisha's feedback from notes for step {step_id}"
                        )
                        return note_content
        except Exception as e:
            logger.warning(f"[TETYANA] Could not retrieve from notes: {e}")

        # Fallback to memory
        try:
            result = await mcp_manager.call_tool(
                "memory", "search_nodes", {"query": f"grisha_rejection_step_{step_id}"}
            )

            if result and hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        logger.info(
                            f"[TETYANA] Retrieved Grisha's feedback from memory for step {step_id}"
                        )
                        return item.text
            elif isinstance(result, dict) and "entities" in result:
                entities = result["entities"]
                if entities and len(entities) > 0:
                    logger.info(
                        f"[TETYANA] Retrieved Grisha's feedback from memory for step {step_id}"
                    )
                    return entities[0].get("observations", [""])[0]

        except Exception as e:
            logger.warning(f"[TETYANA] Could not retrieve Grisha's feedback: {e}")

        return None

    async def _take_screenshot_for_vision(self, pid: int = None) -> Optional[str]:
        """Take screenshot for Vision analysis, optionally focusing on specific app."""
        from ..logger import logger  # noqa: E402
        import subprocess  # noqa: E402
        import base64
        from datetime import datetime  # noqa: E402
        from ..config import SCREENSHOTS_DIR  # noqa: E402

        try:
            # Create screenshots directory if needed
            os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOTS_DIR, f"vision_{timestamp}.png")
            
            # If PID provided, try to focus that app first
            if pid:
                try:
                    focus_script = f'''
                    tell application "System Events"
                        set frontProcess to first process whose unix id is {pid}
                        set frontmost of frontProcess to true
                    end tell
                    '''
                    subprocess.run(["osascript", "-e", focus_script], capture_output=True, timeout=5)
                    await asyncio.sleep(0.3)  # Wait for focus
                except Exception as e:
                    logger.warning(f"[TETYANA] Could not focus app {pid}: {e}")
            
            # 1. Try MCP Tool first (Native Swift)
            try:
                # We need to construct a lightweight call since we are inside Tetyana agent class, 
                # but we have access to mcp_manager via import
                if "macos-use" in mcp_manager.config.get("mcpServers", {}):
                     result = await mcp_manager.call_tool("macos-use", "macos-use_take_screenshot", {})
                     
                     base64_img = None
                     if isinstance(result, dict) and "content" in result:
                         for item in result["content"]:
                             if item.get("type") == "text":
                                 base64_img = item.get("text")
                                 break
                     elif hasattr(result, "content") and len(result.content) > 0:
                          if hasattr(result.content[0], "text"):
                               base64_img = result.content[0].text
                               
                     if base64_img:
                          with open(path, "wb") as f:
                              f.write(base64.b64decode(base64_img))
                          logger.info(f"[TETYANA] Screenshot for Vision saved via MCP: {path}")
                          return path
            except Exception as e:
                logger.warning(f"[TETYANA] MCP screenshot failed, falling back: {e}")

            # 2. Fallback to screencapture
            result = subprocess.run(
                ["screencapture", "-x", path],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0 and os.path.exists(path):
                logger.info(f"[TETYANA] Screenshot for Vision saved (fallback): {path}")
                return path
            else:
                logger.error(f"[TETYANA] Screenshot failed: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"[TETYANA] Screenshot error: {e}")
            return None

    async def analyze_screen(self, query: str, pid: int = None) -> Dict[str, Any]:
        """
        Take screenshot and analyze with Vision to find UI elements.
        Used for complex GUI tasks where Accessibility Tree is insufficient.
        
        Args:
            query: What to look for (e.g., "Find the 'Next' button")
            pid: Optional PID to focus app before screenshot
            
        Returns:
            {"found": bool, "elements": [...], "current_state": str, "suggested_action": {...}}
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
        from ..logger import logger  # noqa: E402
        import base64  # noqa: E402

        logger.info(f"[TETYANA] Vision analysis requested: {query}")
        
        # Use provided PID or tracked PID
        effective_pid = pid or self._current_pid
        
        # 1. Take screenshot
        screenshot_path = await self._take_screenshot_for_vision(effective_pid)
        if not screenshot_path:
            return {"found": False, "error": "Could not take screenshot"}
        
        # 2. Load and encode image
        try:
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return {"found": False, "error": f"Could not read screenshot: {e}"}
        
        # 3. Vision analysis prompt
        vision_prompt = f"""Analyze this macOS screenshot to help with: {query}

You are assisting with GUI automation. Identify clickable elements, their positions, and suggest the best action.

Respond in JSON format:
{{
    "found": true/false,
    "elements": [
        {{
            "type": "button|link|textfield|checkbox|menu",
            "label": "Element text or description",
            "x": 350,
            "y": 420,
            "confidence": 0.95
        }}
    ],
    "current_state": "Brief description of what's visible on screen",
    "suggested_action": {{
        "tool": "macos-use_click_and_traverse",
        "args": {{"pid": {effective_pid or 'null'}, "x": 350, "y": 420}}
    }},
    "notes": "Any important observations (CAPTCHA detected, page loading, etc.)"
}}

IMPORTANT:
- Coordinates should be approximate center of the element
- If you see a CAPTCHA or verification challenge, note it in "notes"
- If the target element is not visible, set "found": false and explain in "current_state"
"""
        
        content = [
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ]
        
        messages = [
            SystemMessage(content="You are a Vision assistant for macOS GUI automation. Analyze screenshots precisely and provide accurate element coordinates."),
            HumanMessage(content=content),
        ]
        
        try:
            response = await self.vision_llm.ainvoke(messages)
            result = self._parse_response(response.content)
            
            if result.get("found"):
                logger.info(f"[TETYANA] Vision found elements: {len(result.get('elements', []))}")
                logger.info(f"[TETYANA] Current state: {result.get('current_state', '')[:100]}...")
            else:
                logger.warning(f"[TETYANA] Vision did not find target: {result.get('current_state', 'Unknown')}")
            
            # Store screenshot path for Grisha verification
            result["screenshot_path"] = screenshot_path
            return result
            
        except Exception as e:
            logger.error(f"[TETYANA] Vision analysis failed: {e}")
            return {"found": False, "error": str(e), "screenshot_path": screenshot_path}

    def _get_dynamic_temperature(self, attempt: int) -> float:
        """Dynamic temperature: 0.1 + attempt * 0.2, capped at 1.0"""
        return min(0.1 + (attempt * 0.2), 1.0)

    async def execute_step(self, step: Dict[str, Any], attempt: int = 1) -> StepResult:
        """
        Executes a single plan step with Advanced Reasoning:
        1. Internal Monologue (Thinking before acting) - SKIPPED for simple tools
        2. Tool Execution
        3. Technical Reflexion (Self-correction on failure) - SKIPPED for transient errors
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

        from ..logger import logger  # noqa: E402
        from ..state_manager import state_manager  # noqa: E402

        self.attempt_count = attempt
        
        # Get step_id early for logging
        step_id = step.get("id", self.current_step)

        # --- SPECIAL CASE: Consent/Approval Steps ---
        # Robust check for provided response in message bus or previous context
        provided_response = None
        if "bus_messages" in step:
            for bm in step["bus_messages"]:
                payload = bm.get("payload", {})
                if "user_response" in payload:
                    provided_response = payload["user_response"]
                    logger.info(f"[TETYANA] Found provided response in bus_messages: {provided_response}")
                    break
        
        # If not in bus_messages, check previous_results for Atlas's autonomous decision
        if not provided_response and "previous_results" in step:
            for pr in reversed(step["previous_results"]):
                if pr.get("error") == "autonomous_decision_made" or pr.get("error") == "user_input_received":
                    # The response should have been injected into message bus, but as a backup check
                    pass

        # Refined consent detection
        step_action_lower = str(step.get("action", "")).lower()
        user_keywords = ["user", "human", "oleg", "me", "operator", "owner", "користувач", "людин", "олег", "мені"]
        technical_user_refs = ["active user", "system user", "user home", "user account", "user directory", "current user", "username"]
        
        # Heuristic to detect if action truly targets the HUMAN operator
        is_human_targeted = any(u in step_action_lower for u in user_keywords)
        if is_human_targeted:
            for tr in technical_user_refs:
                if tr in step_action_lower:
                    is_human_targeted = False
                    break
        
        # Only trigger consent if we DON'T have a response yet
        is_consent_request = (
            (not provided_response) and (
                (any(kw in step_action_lower for kw in ["ask", "request", "await", "get", "confirm"]) and is_human_targeted)
                or ("consent" in step_action_lower)
                or ("approval" in step_action_lower and is_human_targeted)
                or ("confirmation" in step_action_lower and is_human_targeted)
                or ("preferences" in step_action_lower and is_human_targeted)
                or step.get("requires_consent", False) is True
            )
        )

        if is_consent_request:
            logger.info(
                f"[TETYANA] Step '{step_id}' requires consent. Signal orchestrator."
            )
            consent_msg = f"Потрібна ваша згода або відповідь для кроку: {step.get('action')}\nОчікуваний результат: {step.get('expected_result', 'Підтвердження користувача')}"

            return StepResult(
                step_id=step.get("id", self.current_step),
                success=False,
                result=consent_msg,
                voice_message="Мені потрібна ваша згода або додаткова інформація. Будь ласка, напишіть у чат.",
                error="need_user_input",
                thought=f"I detected a need for user consent in step: {step.get('action')}. provided_response={provided_response}.",
            )
        
        if provided_response:
             logger.info(f"[TETYANA] Proceeding with provided response: {provided_response}")
             # We let the reasoning LLM know we have this response via 'bus_messages' which is already in the prompt

        # --- OPTIMIZATION: SMART REASONING GATE ---
        # Skip reasoning LLM for well-defined, simple tools
        # Skip reasoning LLM for well-defined, simple tools
        # We KEEP "terminal" here for speed, but tools like "git" or "macos-use" benefit from reasoning (coordinates, args)
        SKIP_REASONING_TOOLS = ["terminal", "filesystem", "time", "fetch"]
        TRANSIENT_ERRORS = [
            "Connection refused",
            "timeout",
            "rate limit",
            "Broken pipe",
            "Connection reset",
        ]

        # --- PHASE 0: DYNAMIC INSPECTION ---
        actual_step_id = step.get('id', self.current_step)
        logger.info(f"[TETYANA] Executing step {actual_step_id}...")
        context_data = shared_context.to_dict()

        # Populate tools summary if empty
        if not shared_context.available_tools_summary:
            logger.info("[TETYANA] Fetching fresh MCP catalog for context...")
            shared_context.available_tools_summary = await mcp_manager.get_mcp_catalog()

        # --- PHASE 0.5: VISION ANALYSIS (if required) ---
        # When step has requires_vision=true, use Vision to find UI elements
        vision_result = None
        if step.get("requires_vision") and attempt <= 2:
            logger.info("[TETYANA] Step requires Vision analysis for UI element discovery...")
            query = step.get("action", "Find the next interaction target")
            
            # Try to get PID from step args or tracked state
            step_pid = None
            if step.get("args") and isinstance(step.get("args"), dict):
                step_pid = step["args"].get("pid")
            
            vision_result = await self.analyze_screen(query, step_pid or self._current_pid)
            
            if vision_result.get("found") and vision_result.get("suggested_action"):
                suggested = vision_result["suggested_action"]
                logger.info(f"[TETYANA] Vision suggests action: {suggested}")
                
                # If Vision found the element, we can use its suggestion directly
                # This will be used in the tool_call below
            elif vision_result.get("notes"):
                # Check for CAPTCHA or other blockers
                notes = vision_result.get("notes", "").lower()
                if "captcha" in notes or "verification" in notes or "robot" in notes:
                    logger.warning(f"[TETYANA] Vision detected blocker: {vision_result.get('notes')}")
                    return StepResult(
                        step_id=step.get("id", self.current_step),
                        success=False,
                        result="Vision detected CAPTCHA or verification challenge",
                        voice_message="Виявлено CAPTCHA або перевірку. Потрібна допомога користувача.",
                        error=f"Blocker detected: {vision_result.get('notes')}",
                        screenshot_path=vision_result.get("screenshot_path"),
                    )

        vibe_guardrail_msg = ""

        # Fetch Grisha's feedback (Priority: Injected via self-healing loop > Saved rejection report)
        grisha_feedback = step.get("grisha_feedback", "")
        if not grisha_feedback and attempt > 1:
            logger.info(f"[TETYANA] Attempt {attempt} - fetching Grisha's rejection feedback from memory...")
            grisha_feedback = await self.get_grisha_feedback(step.get("id")) or ""
        
        if grisha_feedback:
             logger.info(f"[TETYANA] Improving execution with Grisha's feedback: {grisha_feedback[:100]}...")

        target_server = step.get("realm") or step.get("tool") or step.get("server")
        # Normalize generic 'browser' realm to macos-use to leverage native automation
        if target_server == "browser":
            target_server = "macos-use"
        tools_summary = ""
        monologue = {}

        # SMART GATE: Check if we can skip reasoning
        # Only skip if it's the first attempt; on retries (attempt > 1) or if FEEDBACK exists, always use reasoning
        skip_reasoning = (
            attempt == 1
            and not grisha_feedback
            and target_server in SKIP_REASONING_TOOLS
            and step.get("tool")
            and step.get("args")
            and not step.get("requires_vision")
        )

        if skip_reasoning:
            # Direct execution path - no LLM call needed
            logger.info(
                f"[TETYANA] FAST PATH: Skipping reasoning for simple tool '{target_server}'"
            )
            tool_call = {
                "name": step.get("tool"),
                "args": step.get("args", {}),
                "server": target_server,
            }
        else:
            # Full reasoning path for complex/ambiguous steps
            configured_servers = mcp_manager.config.get("mcpServers", {})
            if target_server in configured_servers and not target_server.startswith("_"):
                logger.info(f"[TETYANA] Dynamically inspecting server: {target_server}")
                tools = await mcp_manager.list_tools(target_server)
                import json  # noqa: E402

                tools_summary = f"\n--- DETAILED SPECS FOR SERVER: {target_server} ---\n"
                for t in tools:
                    name = getattr(t, "name", str(t))
                    desc = getattr(t, "description", "")
                    schema = getattr(t, "inputSchema", {})
                    tools_summary += (
                        f"- {name}: {desc}\n  Schema: {json.dumps(schema, ensure_ascii=False)}\n"
                    )
            else:
                tools_summary = getattr(
                    shared_context,
                    "available_tools_summary",
                    "List available tools using list_tools if needed.",
                )

            # Extract previous_results from step if available
            previous_results = step.get("previous_results")

            reasoning_prompt = AgentPrompts.tetyana_reasoning_prompt(
                str(step),
                context_data,
                tools_summary=tools_summary,
                feedback=grisha_feedback,
                previous_results=previous_results,
                goal_context=shared_context.get_goal_context(),
                bus_messages=step.get("bus_messages"),
                full_plan=step.get("full_plan", ""),
            )

            try:
                reasoning_resp = await self.reasoning_llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are a Technical Executor. Think technically in English about tools and arguments."
                        ),
                        HumanMessage(content=reasoning_prompt),
                    ]
                )
                monologue = self._parse_response(reasoning_resp.content)
                logger.info(
                    f"[TETYANA] Thought (English): {monologue.get('thought', 'No thought')[:200]}..."
                )

                # Check for proactive help request
                if monologue.get("question_to_atlas"):
                    question = monologue["question_to_atlas"]
                    logger.info(f"[TETYANA] Proactive help request to Atlas: {question}")
                    
                    from ..message_bus import AgentMsg, MessageType, message_bus # noqa: E402
                    msg = AgentMsg(
                        from_agent="tetyana",
                        to_agent="atlas",
                        message_type=MessageType.HELP_REQUEST,
                        payload={"question": question, "step_id": step.get("id")},
                        step_id=step.get("id")
                    )
                    await message_bus.send(msg)
                    
                    return StepResult(
                        step_id=step.get("id", self.current_step),
                        success=False,
                        result=f"Blocked on Atlas: {question}",
                        voice_message=monologue.get("voice_message") or f"У мене питання до Атласу: {question}",
                        error="proactive_help_requested",
                        thought=monologue.get("thought")
                    )

                tool_call = (
                    monologue.get("proposed_action")
                    or step.get("tool_call")
                    or {
                        "name": step.get("tool") or "",
                        "args": step.get("args")
                        or {"action": step.get("action"), "path": step.get("path")},
                    }
                )

                if not tool_call.get("name"):
                    # Enhanced fallback: Try to infer tool name from step metadata
                    inferred_name = (
                        step.get("tool") or
                        step.get("server") or
                        step.get("realm")
                    )
                    # Try action-based inference
                    if not inferred_name:
                        action_text = str(step.get("action", "")).lower()
                        if any(kw in action_text for kw in ["implement feature", "deep code", "refactor project"]):
                            inferred_name = "vibe_implement_feature"
                        elif any(kw in action_text for kw in ["vibe", "code", "debug", "analyze error"]):
                            inferred_name = "vibe_prompt"
                        elif any(kw in action_text for kw in ["click", "type", "press", "scroll", "open app"]):
                            inferred_name = "macos-use_take_screenshot" # Default to screenshot if UI action
                        elif any(kw in action_text for kw in ["finder", "desktop", "folder", "sort", "trash", "open path"]):
                            inferred_name = "macos-use_finder_list_files"
                        elif any(kw in action_text for kw in ["read_file", "write_file", "list_directory"]):
                            inferred_name = "filesystem" 
                        elif any(kw in action_text for kw in ["run", "execute", "command", "terminal", "bash", "mkdir"]):
                            inferred_name = "execute_command" 
                    
                    if inferred_name:
                        tool_call["name"] = inferred_name
                        logger.info(f"[TETYANA] Inferred tool name from step metadata: {inferred_name}")
                    else:
                        logger.warning("[TETYANA] LLM monologue missing 'proposed_action'. Could not infer tool name.")
                
                # VISION OVERRIDE: If Vision found the element with high confidence, use its suggestion
                if vision_result and vision_result.get("found") and vision_result.get("suggested_action"):
                    suggested = vision_result["suggested_action"]
                    # Merge Vision's coordinates into tool_call args
                    if suggested.get("args"):
                        if not isinstance(tool_call.get("args"), dict):
                            tool_call["args"] = {}
                        # Update with Vision-provided coordinates
                        for key in ["x", "y", "pid"]:
                            if key in suggested["args"] and suggested["args"][key] is not None:
                                tool_call["args"][key] = suggested["args"][key]
                                logger.info(f"[TETYANA] Vision override: {key}={suggested['args'][key]}")
                        # If Vision suggests a specific tool, consider using it
                        if suggested.get("tool") and "click" in suggested["tool"].lower():
                            tool_call["name"] = suggested["tool"]
                            tool_call["server"] = "macos-use"
                            logger.info(f"[TETYANA] Vision override: tool={suggested['tool']}")
            except Exception as e:
                logger.warning(f"[TETYANA] Internal Monologue failed: {e}")
                tool_call = {
                    "name": step.get("tool"),
                    "args": {"action": step.get("action"), "path": step.get("path")},
                }

        if target_server and "server" not in tool_call:
            tool_call["server"] = target_server

        # Attach step id to args when available so wrappers can collect artifacts
        try:
            if isinstance(tool_call.get("args"), dict):
                tool_call["args"]["step_id"] = step.get("id")
        except Exception:
            pass

        # --- AUTO-FILL PID FOR MACOS-USE TOOLS ---
        # If this is a macos-use tool and pid is missing, use tracked _current_pid
        tool_name_lower = str(tool_call.get("name", "")).lower()
        tool_server = tool_call.get("server", "")
        if tool_name_lower.startswith("macos-use") or tool_server == "macos-use":
            args = tool_call.get("args", {})
            if isinstance(args, dict) and not args.get("pid") and self._current_pid:
                tool_call.setdefault("args", {})["pid"] = self._current_pid
                logger.info(f"[TETYANA] Auto-filled pid from tracked state: {self._current_pid}")

        # --- PHASE 2: TOOL EXECUTION ---
        tool_result = await self._execute_tool(tool_call)

        # --- PHASE 3: TECHNICAL REFLEXION (if failed) ---
        # OPTIMIZATION: Skip LLM reflexion for transient errors
        max_self_fixes = 3
        fix_count = 0

        while not tool_result.get("success") and fix_count < max_self_fixes:
            fix_count += 1
            error_msg = tool_result.get("error", "Unknown error")

            # Check for transient errors - simple retry without LLM
            is_transient = any(err.lower() in error_msg.lower() for err in TRANSIENT_ERRORS)

            if is_transient:
                logger.info(
                    f"[TETYANA] Transient error detected. Quick retry {fix_count}/{max_self_fixes}..."
                )
                await asyncio.sleep(1.0 * fix_count)  # Simple backoff
                tool_result = await self._execute_tool(tool_call)
                if tool_result.get("success"):
                    logger.info("[TETYANA] Quick retry SUCCESS!")
                    break
                continue

            # Full reflexion for logic/permission errors
            logger.info(
                f"[TETYANA] Step failed. Reflexion Attempt {fix_count}/{max_self_fixes}. Error: {error_msg}"
            )

            # ULTIMATE FIX: Invoke VIBE for deep healing on final attempts
            if fix_count == max_self_fixes:
                 logger.info("[TETYANA] Reflexion limit reached. Invoking VIBE for ultimate self-healing...")
                 v_res_raw = await self._call_mcp_direct("vibe", "vibe_analyze_error", {
                     "error_message": error_msg,
                     "auto_fix": True
                 })
                 # Convert CallToolResult to dict using existing formatter
                 v_res = self._format_mcp_result(v_res_raw) if v_res_raw else {}
                 
                 # Check if vibe fixed it
                 if v_res.get("success"):
                      logger.info("[TETYANA] VIBE self-healing reported SUCCESS. Retrying original tool...")
                      tool_result = await self._execute_tool(tool_call)
                      if tool_result.get("success"):
                          break
                 else:
                      logger.warning(f"[TETYANA] VIBE self-healing failed: {v_res.get('error')}")

            try:
                tools_summary = (
                    hasattr(shared_context, "available_tools_summary")
                    and shared_context.available_tools_summary
                    or ""
                )
                reflexion_prompt = AgentPrompts.tetyana_reflexion_prompt(
                    str(step),
                    error_msg,
                    [r.to_dict() for r in self.results[-5:]],
                    tools_summary=tools_summary,
                )

                reflexion_resp = await self.reflexion_llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are a Technical Debugger. Analyze the tool error in English and suggest a fix."
                        ),
                        HumanMessage(content=reflexion_prompt),
                    ]
                )

                reflexion = self._parse_response(reflexion_resp.content)

                if reflexion.get("requires_atlas"):
                    logger.info("[TETYANA] Reflexion determined Atlas intervention is required.")
                    break

                fix_action = reflexion.get("fix_attempt")
                if not fix_action:
                    break

                logger.info(f"[TETYANA] Attempting autonomous fix: {fix_action.get('tool')}")
                tool_result = await self._execute_tool(fix_action)

                if tool_result.get("success"):
                    logger.info("[TETYANA] Autonomous fix SUCCESS!")
                    break
            except Exception as re:
                logger.error(f"[TETYANA] Reflexion failed: {re}")
                break

        voice_msg = tool_result.get("voice_message") or (
            monologue.get("voice_message") if attempt == 1 else None
        )

        # Fallback if no specific voice message from LLM/Tool
        if not voice_msg and attempt == 1:
            voice_msg = self.get_voice_message(
                "completed" if tool_result.get("success") else "failed",
                step=step.get("id", self.current_step),
                description=step.get("action", ""),
            )

        res = StepResult(
            step_id=step.get("id", self.current_step),
            success=tool_result.get("success", False),
            result=tool_result.get("output", ""),
            screenshot_path=tool_result.get("screenshot_path") or (vision_result.get("screenshot_path") if vision_result else None),
            voice_message=voice_msg,
            error=tool_result.get("error"),
            tool_call=tool_call,
            thought=monologue.get("thought") if isinstance(monologue, dict) else None,
        )

        self.results.append(res)

        # Update current step counter
        step_id = step.get("id", 0)
        try:
            self.current_step = int(step_id) + 1
        except Exception:
            self.current_step += 1

        if state_manager.available:
            state_manager.checkpoint("current", res.step_id, res.to_dict())

        return res

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the tool call via unified Dispatcher"""
        from ..mcp_manager import mcp_manager
        
        # Ensure dispatcher knows about current PID
        mcp_manager.dispatcher.set_pid(self._current_pid)
        
        tool_name = tool_call.get("name") or tool_call.get("tool")
        args = tool_call.get("args") or tool_call.get("arguments") or {}
        explicit_server = tool_call.get("server")

        try:
            # Use unified dispatch_tool
            result = await mcp_manager.dispatch_tool(tool_name, args, explicit_server)
            
            # Normalize Pydantic object to dict
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            elif hasattr(result, "dict"): # Fallback for older Pydantic
                result = result.dict()
            elif not isinstance(result, dict):
                # Handle unexpected types (list, primitive)
                result = {"content": [{"type": "text", "text": str(result)}], "isError": False}

            # Normalize success/error flags
            # MCP SDK uses 'isError' (bool), we use 'success' (bool) and 'error' (str)
            if isinstance(result, dict):
                if "isError" in result:
                    result["success"] = not result["isError"]
                
                # Ensure 'success' key exists
                if "success" not in result:
                    # Assume success if no error field
                    result["success"] = "error" not in result
                
                # If failed, ensure 'error' field exists for feedback loop
                if not result["success"] and "error" not in result:
                    # Extract error from content if possible
                    content_text = ""
                    if "content" in result and isinstance(result["content"], list):
                        for item in result["content"]:
                            if item.get("type") == "text":
                                content_text += item.get("text", "")
                    result["error"] = content_text or "Unknown tool execution error"

            return result


        except Exception as e:
            logger.error(f"[TETYANA] Tool execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _call_mcp_direct(self, server: str, tool: str, args: Dict) -> Dict[str, Any]:
        from ..mcp_manager import mcp_manager
        try:
            return await mcp_manager.dispatch_tool(tool, args, server)
        except Exception as e:
            logger.error(f"[TETYANA] Unified call failed for {server}.{tool}: {e}")
            return {"success": False, "error": str(e)}

    async def _run_terminal_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a bash command using Terminal MCP"""
        import re  # noqa: E402

        from ..mcp_manager import mcp_manager  # noqa: E402

        command = args.get("command", "") or args.get("cmd", "") or ""

        # SAFETY CHECK: Block Cyrillic characters
        if re.search(r"[а-яА-ЯіїєґІЇЄҐ]", command):
            return {
                "success": False,
                "error": f"Command blocked: Contains Cyrillic characters. You are trying to execute a description instead of a command: '{command}'",
            }

        # Pass all args to the tool (supports cwd, stdout_file, etc.)
        # OPTIMIZATION: Use 'macos-use' server which now handles terminal commands natively
        res = await mcp_manager.call_tool("macos-use", "execute_command", args)
        return self._format_mcp_result(res)

    async def _perform_gui_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Performs GUI interaction (click, type, hotkey, search_app) using pyautogui"""
        try:
            import pyautogui  # noqa: E402

            from ..mcp_manager import mcp_manager  # noqa: E402

            action = args.get("action", "")

            if action == "click":
                x, y = args.get("x", 0), args.get("y", 0)
                pid = int(args.get("pid", 0))
                res = await mcp_manager.call_tool(
                    "macos-use",
                    "macos-use_click_and_traverse",
                    {"pid": pid, "x": float(x), "y": float(y)},
                )
                return self._format_mcp_result(res)

            elif action == "type":
                text = args.get("text", "")
                pid = int(args.get("pid", 0))
                res = await mcp_manager.call_tool(
                    "macos-use", "macos-use_type_and_traverse", {"pid": pid, "text": text}
                )
                return self._format_mcp_result(res)

            elif action == "hotkey":
                keys = args.get("keys", [])
                pid = int(args.get("pid", 0))

                # Mapper for Swift SDK keys
                modifiers = []
                key_name = ""

                modifier_map = {
                    "cmd": "Command",
                    "command": "Command",
                    "shift": "Shift",
                    "ctrl": "Control",
                    "control": "Control",
                    "opt": "Option",
                    "option": "Option",
                    "alt": "Option",
                    "fn": "Function",
                }

                for k in keys:
                    lower_k = k.lower()
                    if lower_k in modifier_map:
                        modifiers.append(modifier_map[lower_k])
                    else:
                        # Key Map
                        key_map = {
                            "enter": "Return",
                            "return": "Return",
                            "esc": "Escape",
                            "escape": "Escape",
                            "space": "Space",
                            "tab": "Tab",
                            "up": "ArrowUp",
                            "down": "ArrowDown",
                            "left": "ArrowLeft",
                            "right": "ArrowRight",
                            "delete": "Delete",
                            "backspace": "Delete",
                            "home": "Home",
                            "end": "End",
                            "pageup": "PageUp",
                            "pagedown": "PageDown",
                            "f1": "F1", "f2": "F2", "f3": "F3", "f4": "F4", "f5": "F5",
                            "f6": "F6", "f7": "F7", "f8": "F8", "f9": "F9", "f10": "F10",
                            "f11": "F11", "f12": "F12",
                        }
                        key_name = key_map.get(lower_k, k)  # Default to raw key (e.g. "a", "1")

                if not key_name and not modifiers:
                    return {"success": False, "error": "Invalid hotkey definition"}

                # If only modifiers, we can't really "press" a key in this API, needs a key
                if not key_name:
                    # Fallback or error? Let's error for now
                    return {"success": False, "error": "No non-modifier key specified"}

                res = await mcp_manager.call_tool(
                    "macos-use",
                    "macos-use_press_key_and_traverse",
                    {"pid": pid, "keyName": key_name, "modifierFlags": modifiers},
                )
                return self._format_mcp_result(res)

            elif action == "wait" or action == "sleep":
                duration = float(args.get("duration", 1.0))
                time.sleep(duration)
                return {"success": True, "output": f"Waited for {duration} seconds"}

            elif action == "search_app":
                app_name = args.get("app_name", "") or args.get("text", "")
                import subprocess  # noqa: E402

                # 0. Try robust macos-use_open_application_and_traverse first
                # This gives us the PID and Accessibility Tree, which is superior to just opening
                try:
                    logger.info(f"[TETYANA] Attempting to open '{app_name}' via macos-use native tool...")
                    res = await mcp_manager.call_tool(
                        "macos-use",
                        "macos-use_open_application_and_traverse",
                        {"identifier": app_name},
                    )
                    # Helper to check if MCP call was actually successful
                    formatted = self._format_mcp_result(res)
                    if formatted.get("success") and not formatted.get("error"):
                        logger.info(f"[TETYANA] Successfully opened '{app_name}' via macos-use.")
                        return formatted
                    else:
                        logger.warning(
                            f"[TETYANA] macos-use open failed, falling back to legacy: {formatted.get('error')}"
                        )
                except Exception as e:
                    logger.warning(f"[TETYANA] macos-use open exception: {e}")

                # 1. Try 'open -a'
                try:
                    if app_name.lower() in ["calculator", "калькулятор"]:
                        app_name = "Calculator"
                    subprocess.run(["open", "-a", app_name], check=True, capture_output=True)
                    return {"success": True, "output": f"Launched app: {app_name}"}
                except Exception:
                    pass

                # 2. Spotlight fallback
                # Try to force English layout (ABC/U.S./English)
                switch_script = """
                tell application "System Events"
                    try
                        tell process "SystemUIServer"
                            set input_menu to (menu bar items of menu bar 1 whose description is "text input")
                            if (count of input_menu) > 0 then
                                click item 1 of input_menu
                                delay 0.2
                                set menu_items to menu 1 of item 1 of input_menu
                                repeat with mi in menu_items
                                    set mname to name of mi
                                    if mname is "ABC" or mname is "U.S." or mname is "English" or mname is "British" then
                                        click mi
                                        exit repeat
                                    end if
                                end repeat
                            end if
                        end tell
                    on error err
                        log err
                    end try
                end tell
                """
                subprocess.run(["osascript", "-e", switch_script], capture_output=True)

                # Open Spotlight (Command+Space)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to key code 49 using {command down}',
                    ],
                    check=True,
                )
                time.sleep(1.0)  # Wait for Spotlight to appear

                # Copy app name to clipboard
                process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                process.communicate(input=app_name.encode("utf-8"))

                # Clear search field (Cmd+A, Backspace)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to key code 0 using {command down}',
                    ],
                    check=True,
                )
                time.sleep(0.2)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to key code 51',
                    ],
                    check=True,
                )
                time.sleep(0.2)

                # Paste using Command+V (Key code 9)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to key code 9 using {command down}',
                    ],
                    check=True,
                )
                time.sleep(0.5)
                # Press Enter (Key code 36)
                subprocess.run(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to key code 36',
                    ],
                    check=True,
                )
                return {
                    "success": True,
                    "output": f"Launched app via Spotlight (Clipboard Paste): {app_name}",
                }
            else:
                return {"success": False, "error": f"Unknown GUI action: {action}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _browser_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Browser action via Puppeteer MCP

        Additionally collects verification artifacts (HTML, screenshot, small evaluations)
        and persists them to disk/notes/memory so that Grisha can verify actions."""
        import base64  # noqa: E402
        import time as _time  # noqa: E402

        from ..config import SCREENSHOTS_DIR, WORKSPACE_DIR  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        action = args.get("action", "")
        step_id = args.get("step_id")

        # Helper: save artifact files and register in notes/memory
        async def _save_artifacts(
            html_text: str = None, title_text: str = None, screenshot_b64: str = None
        ):
            try:
                # from ..config import SCREENSHOTS_DIR, WORKSPACE_DIR  # noqa: E402
                from ..logger import logger  # noqa: E402

                # from ..mcp_manager import mcp_manager  # noqa: E402

                ts = _time.strftime("%Y%m%d_%H%M%S")
                artifacts = []

                if html_text:
                    html_file = WORKSPACE_DIR / f"grisha_step_{step_id}_{ts}.html"
                    html_file.parent.mkdir(parents=True, exist_ok=True)
                    html_file.write_text(html_text, encoding="utf-8")
                    artifacts.append(str(html_file))
                    logger.info(f"[TETYANA] Saved HTML artifact: {html_file}")

                if screenshot_b64:
                    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                    img_file = SCREENSHOTS_DIR / f"grisha_step_{step_id}_{ts}.png"
                    with open(img_file, "wb") as f:
                        f.write(base64.b64decode(screenshot_b64))
                    artifacts.append(str(img_file))
                    logger.info(f"[TETYANA] Saved screenshot artifact: {img_file}")

                # Create a note linking artifacts and include short HTML/title snippet and keyword checks
                note_title = f"Grisha Artifact - Step {step_id} @ {ts}"
                snippet = ""
                if title_text:
                    snippet += f"Title: {title_text}\n\n"
                if html_text:
                    snippet += f"HTML Snippet:\n{(html_text[:1000] + '...') if len(html_text) > 1000 else html_text}\n\n"

                # Keyword search in HTML snippet
                detected = []
                if html_text:
                    keywords = ["phone", "sms", "verification", "код", "телефон"]
                    low = html_text.lower()
                    for kw in keywords:
                        if kw in low:
                            detected.append(kw)

                note_content = (
                    f"Artifacts for step {step_id} saved at {ts}.\n\nFiles:\n"
                    + ("\n".join(artifacts) if artifacts else "(no binary files captured)")
                    + "\n\n"
                    + snippet
                )
                if detected:
                    note_content += f"Detected keywords in HTML: {', '.join(detected)}\n"

                try:
                    await mcp_manager.dispatch_tool(
                        "notes_create",
                        {
                            "body": f"# {note_title}\n\n{note_content}"
                        },
                    )
                    logger.info(f"[TETYANA] Created verification artifact note for step {step_id}")
                except Exception as e:
                    logger.warning(f"[TETYANA] Failed to create artifact note: {e}")

                # Create memory entity for easy search
                try:
                    await mcp_manager.call_tool(
                        "memory",
                        "create_entities",
                        {
                            "entities": [
                                {
                                    "name": f"grisha_artifact_step_{step_id}",
                                    "entityType": "artifact",
                                    "observations": artifacts,
                                }
                            ]
                        },
                    )
                    logger.info(f"[TETYANA] Created memory artifact for step {step_id}")
                except Exception as e:
                    logger.warning(f"[TETYANA] Failed to create memory artifact: {e}")

                return True
            except Exception as e:
                from ..logger import logger  # noqa: E402

                logger.warning(f"[TETYANA] _save_artifacts exception: {e}")

        if action == "navigate" or action == "open":
            res = await mcp_manager.call_tool(
                "puppeteer", "puppeteer_navigate", {"url": args.get("url", "")}
            )

            # Try to collect artifacts (title, html, screenshot)
            try:
                # small delay to allow navigation/rendering
                await asyncio.sleep(1.5)
                from ..logger import logger  # noqa: E402

                logger.info(f"[TETYANA] Collecting browser artifacts for step {step_id}...")

                # Document title
                title_res = await mcp_manager.call_tool(
                    "puppeteer", "puppeteer_evaluate", {"script": "document.title"}
                )
                title_text = None
                if (
                    hasattr(title_res, "content")
                    and len(title_res.content) > 0
                    and hasattr(title_res.content[0], "text")
                ):
                    title_text = title_res.content[0].text

                # Page HTML
                html_res = await mcp_manager.call_tool(
                    "puppeteer",
                    "puppeteer_evaluate",
                    {"script": "document.documentElement.outerHTML"},
                )
                html_text = None
                if (
                    hasattr(html_res, "content")
                    and len(html_res.content) > 0
                    and hasattr(html_res.content[0], "text")
                ):
                    # The evaluation may return a JSON wrapper, try to extract raw
                    html_text = html_res.content[0].text

                # Screenshot (base64)
                shot_res = await mcp_manager.call_tool(
                    "puppeteer",
                    "puppeteer_screenshot",
                    {"name": f"grisha_step_{step_id}", "encoded": True},
                )
                screenshot_b64 = None
                if hasattr(shot_res, "content"):
                    for c in shot_res.content:
                        if getattr(c, "type", "") == "image" and hasattr(c, "data"):
                            screenshot_b64 = c.data
                            break
                        if hasattr(c, "text") and c.text:
                            txt = c.text.strip()
                            # plain base64 or data URI
                            if txt.startswith("iVBOR"):
                                screenshot_b64 = txt
                                break
                            if "base64," in txt:
                                try:
                                    screenshot_b64 = txt.split("base64,", 1)[1]
                                    break
                                except Exception:
                                    pass

                await _save_artifacts(
                    html_text=html_text, title_text=title_text, screenshot_b64=screenshot_b64
                )
            except Exception as e:
                from ..logger import logger  # noqa: E402

                logger.warning(f"[TETYANA] Failed to collect browser artifacts: {e}")

            return self._format_mcp_result(res)
        elif action == "click":
            res = await mcp_manager.call_tool(
                "puppeteer",
                "puppeteer_click",
                {"selector": args.get("selector", "")},
            )

            # If click likely submitted a form, collect artifacts as well
            selector = args.get("selector", "") or ""
            if any(k in selector.lower() for k in ["submit", "next", "confirm", "phone", "sms"]):
                try:
                    # small delay to allow navigation
                    await asyncio.sleep(1.0)
                    # reuse collection
                    await self._browser_action(
                        {"action": "navigate", "url": args.get("url", ""), "step_id": step_id}
                    )
                except Exception:
                    pass

            return self._format_mcp_result(res)
        elif action == "type" or action == "fill":
            return self._format_mcp_result(
                await mcp_manager.call_tool(
                    "puppeteer",
                    "puppeteer_fill",
                    {"selector": args.get("selector", ""), "value": args.get("value", "")},
                )
            )

            return self._format_mcp_result(
                await mcp_manager.call_tool(
                    "puppeteer",
                    "puppeteer_fill",
                    {
                        "selector": args.get("selector", ""),
                        "value": args.get("text", ""),
                    },
                )
            )
        elif action == "screenshot":
            return self._format_mcp_result(
                await mcp_manager.call_tool("puppeteer", "puppeteer_screenshot", {})
            )
        else:
            return {"success": False, "error": f"Unknown browser action: {action}"}

    async def _filesystem_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Filesystem operations via MCP"""
        from ..logger import logger  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        action = args.get("action", "")
        path = args.get("path", "")

        # SMART ACTION INFERENCE if action is missing
        if not action:
            if "content" in args:
                action = "write_file"
            elif path.endswith("/") or "." not in path.split("/")[-1]:
                action = "list_directory"
            elif path:
                action = "read_file"
            logger.info(f"[TETYANA] Inferred FS action: {action} for path: {path}")

        if action == "read" or action == "read_file":
            result = await mcp_manager.call_tool("filesystem", "read_file", {"path": path})
            shared_context.update_path(path, "read")
            return self._format_mcp_result(result)
        elif action == "write" or action == "write_file":
            result = await mcp_manager.call_tool(
                "filesystem",
                "write_file",
                {"path": path, "content": args.get("content", "")},
            )
            shared_context.update_path(path, "write")
            return self._format_mcp_result(result)
        elif action == "create_dir" or action == "mkdir" or action == "create_directory":
            result = await mcp_manager.call_tool("filesystem", "create_directory", {"path": path})
            shared_context.update_path(path, "create_directory")
            return self._format_mcp_result(result)
        elif action == "list" or action == "list_directory":
            result = await mcp_manager.call_tool("filesystem", "list_directory", {"path": path})
            shared_context.update_path(path, "access")
            return self._format_mcp_result(result)
        else:
            return {
                "success": False,
                "error": f"Unknown FS action: {action}. Valid: read_file, write_file, list_directory",
            }

    async def _github_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """GitHub actions"""
        from ..mcp_manager import mcp_manager  # noqa: E402

        # Pass-through mostly
        mcp_tool = args.get("tool_name", "search_repositories")
        gh_args = args.copy()
        if "tool_name" in gh_args:
            del gh_args["tool_name"]
        res = await mcp_manager.call_tool("github", mcp_tool, gh_args)
        return self._format_mcp_result(res)

    async def _applescript_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from ..mcp_manager import mcp_manager  # noqa: E402

        action = args.get("action", "execute_script")
        if action == "execute_script":
            return self._format_mcp_result(
                await mcp_manager.call_tool(
                    "applescript", "execute_script", {"script": args.get("script", "")}
                )
            )
        elif action == "open_app":
            return self._format_mcp_result(
                await mcp_manager.call_tool(
                    "applescript",
                    "open_app_safely",
                    {"app_name": args.get("app_name", "")},
                )
            )
        elif action == "volume":
            return self._format_mcp_result(
                await mcp_manager.call_tool(
                    "applescript", "set_system_volume", {"level": args.get("level", 50)}
                )
            )
        return {"success": False, "error": "Unknown applescript action"}

    def _format_mcp_result(self, res: Any) -> Dict[str, Any]:
        """Standardize MCP response to StepResult format"""
        if isinstance(res, dict) and "error" in res:
            return {"success": False, "error": res["error"]}

        output = ""
        if hasattr(res, "content"):
            for item in res.content:
                if hasattr(item, "text"):
                    output += item.text
        elif isinstance(res, dict) and "content" in res:
            for item in res["content"]:
                if isinstance(item, dict):
                    output += item.get("text", "")
                elif hasattr(item, "text"):
                    output += item.text

        # SMART ERROR DETECTION: Often MCP returns success but output contains "Error"
        lower_output = output.lower()
        error_keywords = [
            "error:",
            "failed:",
            "not found",
            "does not exist",
            "denied",
            "permission error",
        ]
        is_error = any(kw in lower_output for kw in error_keywords)

        if (
            is_error and len(output) < 500
        ):  # Don't trigger if it's a huge log that happens to have "error"
            return {"success": False, "error": output, "output": output}

        return {"success": True, "output": output or "Success (No output)"}

    def get_voice_message(self, action: str, **kwargs) -> str:
        """
        Generates context-aware TTS message dynamically.
        """
        # 1. Use LLM-provided message if available (Highest Priority)
        voice_msg = kwargs.get("voice_message")
        if voice_msg and len(voice_msg) > 5:
            return voice_msg

        # 2. Dynamic generation based on context
        step_id = kwargs.get("step", 1)
        desc = kwargs.get("description", "")
        error = kwargs.get("error", "")

        # Extract "essence" of description (first 5-7 words usually contain the verb and object)
        import re  # noqa: E402

        essence = desc
        if len(desc) > 60:
            # Take start, cut at punctuation or reasonable length
            match = re.search(r"^(.{10,50})[.;,]", desc)
            if match:
                essence = match.group(1)
            else:
                essence = desc[:50] + "..."

        # Translate commonly used technical prefixes and terms
        essence = essence.lower()
        
        # 1. Action Mapping (Verbs)
        translations = {
            "create": "Створюю",
            "update": "Оновлюю",
            "check": "Перевіряю",
            "install": "Встановлюю",
            "run": "Запускаю",
            "execute": "Виконую",
            "call": "Викликаю",
            "search": "Шукаю",
            "list": "Переглядаю",
            "read": "Читаю",
            "write": "Записую",
            "delete": "Видаляю",
            "find": "Шукаю",
            "open": "Відкриваю",
            "take": "Роблю",
            "analyze": "Аналізую",
            "confirm": "Підтверджую",
            "verify": "Верифікую"
        }
        
        for eng, ukr in translations.items():
            if essence.startswith(eng):
                essence = essence.replace(eng, ukr, 1)
                break

        # 2. Key Object Mapping (Nouns)
        vocabulary = {
            "filesystem": "файлову систему",
            "directory": "папку",
            "directories": "папки",
            "folder": "папку",
            "folders": "папки",
            "file": "файл",
            "files": "файли",
            "desktop": "робочий стіл",
            "allowed": "дозволені",
            "terminal": "термінал",
            "screenshot": "знімок екрана",
            "screen": "екран",
            "notes": "нотатки",
            "note": "нотатку",
            "calendar": "календар",
            "reminder": "нагадування",
            "reminders": "нагадування",
            "mail": "пошту",
            "email": "пошту",
            "notification": "сповіщення",
            "element": "елемент",
            "elements": "елементи",
            "application": "програму",
            "apps": "програми",
            "browser": "браузер",
            "path": "шлях",
            "contents": "вміст",
            "confirm": "підтвердити",
            "resolve": "визначити",
            "expand": "розгорнути",
            "home": "домашній",
            "canonical": "канонічний",
            "active": "активного",
            "task": "завдання",
            "plan": "план",
            "steps": "кроки",
            "step": "крок",
            "all": "усі",
            "including": "включаючи",
            "hidden": "приховані",
            "items": "елементи",
            "item": "елемент",
            "total": "загальну",
            "number": "кількість",
            "count": "підрахувати",
            "and": "та",
            "with": "з",
            "for": "для",
            "in": "в",
            "on": "на",
            "to": "до",
            "of": " ",
            "the": " ",
            "a": " ",
            "an": " ",
            "list": "список",
            "listed": "перелічені",
            "reading": "читаю",
            "writing": "записую",
            "user": "користувача"
        }

        words = essence.split()
        translated_words = []
        for word in words:
            clean_word = word.strip(".,()[]{}'\"$").lower()
            if clean_word in vocabulary:
                val = vocabulary[clean_word]
                if val.strip():
                    translated_words.append(val)
            elif clean_word in ["$home", "home"]:
                translated_words.append("домашню папку")
            elif len(clean_word) > 1 and all(c in "0123456789abcdefABCDEF-/" for c in clean_word):
                # Skip UUIDs, paths or hex ids which sound bad in TTS
                continue
            else:
                translated_words.append(word)
        
        essence = " ".join(translated_words)
        
        # SECONDARY FILTER: REMOVE ANY WORDS THAT STILL CONTAIN ENGLISH CHARACTERS
        # This prevents technical jargon like "vibe_server" or "json" from leaking into TTS.
        import re
        essence = " ".join([w for w in essence.split() if not re.search(r'[a-zA-Z]', w)])
        
        # If essence becomes empty after filtering, use a generic fallback based on action
        if not essence.strip():
            if action == "starting": essence = "виконую заплановану дію"
            elif action == "completed": essence = "дію завершено"
            else: essence = "поточний етап"

        # Clean up punctuation and spacing
        essence = re.sub(r'\s+', ' ', essence).strip()

        # Construct message based on state
        if action == "completed":
            return f"Крок {step_id}: {essence} — виконано."
        elif action == "failed":
            err_essence = "Помилка."
            if error:
                # Clean error message
                err_clean = str(error).split("\n")[0][:50]
                err_essence = f"Помилка: {err_clean}"
            return f"У кроці {step_id} не вдалося {essence}. {err_essence}"
        elif action == "starting":
            return f"Розпочинаю крок {step_id}: {essence}."
        elif action == "asking_verification":
            return f"Крок {step_id} завершено. Гріша, верифікуй."

        return f"Статус кроку {step_id}: {action}."

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with GitHub API fallback."""
        # Handle GitHub API timeout specially
        if "HTTPSConnectionPool" in content and (
            "Read timed out" in content or "COPILOT ERROR" in content
        ):
            return {
                "tool_call": {
                    "name": "browser",
                    "args": {"action": "navigate", "url": "https://1337x.to"},
                },
                "voice_message": "GitHub API не відповідає, використаю браузер напряму для пошуку.",
            }
        # Use base class parsing
        return super()._parse_response(content)

