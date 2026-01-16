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


@dataclass
class StepResult:
    """Result of step execution"""

    step_id: int
    success: bool
    result: str
    screenshot_path: Optional[str] = None
    voice_message: Optional[str] = None
    error: Optional[str] = None
    tool_call: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    thought: Optional[str] = None

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


class Tetyana:
    """
    Tetyana - The Executor

    Functions:
    - Executing atomic plan steps
    - Interacting with macOS (GUI/Terminal/Apps)
    - Progress reporting
    - Asking Atlas for help when stuck
    """

    # MACOS-USE TOOL SCHEMAS (Swift Binary Priority)
    # Ensures 100% correct argument handling for native macOS automation
    MACOS_USE_SCHEMAS = {
        "macos-use_open_application_and_traverse": {
            "required": ["identifier"],
            "types": {"identifier": str},
        },
        "macos-use_click_and_traverse": {
            "required": ["x", "y"],
            "optional": ["pid", "showAnimation", "animationDuration"],
            "types": {"pid": int, "x": (int, float), "y": (int, float), "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_type_and_traverse": {
            "required": ["text"],
            "optional": ["pid", "showAnimation", "animationDuration"],
            "types": {"pid": int, "text": str, "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_press_key_and_traverse": {
            "required": ["keyName"],
            "optional": ["pid", "modifierFlags", "showAnimation", "animationDuration"],
            "types": {"pid": int, "keyName": str, "modifierFlags": list, "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_refresh_traversal": {
            "required": [],
            "optional": ["pid"],
            "types": {"pid": int},
        },
        "execute_command": {
            "required": ["command"],
            "types": {"command": str},
        },
        "macos-use_take_screenshot": {
            "required": [],
            "types": {},
        },
        "screenshot": {
            "required": [],
            "types": {},
        },
        "macos-use_analyze_screen": {
            "required": [],
            "types": {},
        },
        "vision": {
            "required": [],
            "types": {},
        },
        "ocr": {
            "required": [],
            "types": {},
        },
        "analyze": {
            "required": [],
            "types": {},
        },
        "terminal": {
            "required": ["command"],
            "types": {"command": str},
        },
        "run_command": {
            "required": ["command"],
            "types": {"command": str},
        },
        "macos-use_scroll_and_traverse": {
            "required": ["direction"],
            "optional": ["pid", "amount", "showAnimation", "animationDuration"],
            "types": {"pid": int, "direction": str, "amount": (int, float), "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_right_click_and_traverse": {
            "required": ["x", "y"],
            "optional": ["pid", "showAnimation", "animationDuration"],
            "types": {"pid": int, "x": (int, float), "y": (int, float), "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_double_click_and_traverse": {
            "required": ["x", "y"],
            "optional": ["pid", "showAnimation", "animationDuration"],
            "types": {"pid": int, "x": (int, float), "y": (int, float), "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_drag_and_drop_and_traverse": {
            "required": ["startX", "startY", "endX", "endY"],
            "optional": ["pid", "showAnimation", "animationDuration"],
            "types": {"pid": int, "startX": (int, float), "startY": (int, float), "endX": (int, float), "endY": (int, float), "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_window_management": {
            "required": ["action"],
            "optional": ["pid", "x", "y", "width", "height"],
            "types": {"pid": int, "action": str, "x": (int, float), "y": (int, float), "width": (int, float), "height": (int, float)},
        },
        "scroll": {
            "required": ["direction"],
            "optional": ["pid", "amount"],
            "types": {"pid": int, "direction": str, "amount": (int, float)},
        },
        "right_click": {
            "required": ["x", "y"],
            "optional": ["pid"],
            "types": {"pid": int, "x": (int, float), "y": (int, float)},
        },
        "double_click": {
            "required": ["x", "y"],
            "optional": ["pid"],
            "types": {"pid": int, "x": (int, float), "y": (int, float)},
        },
        "drag_drop": {
            "required": ["startX", "startY", "endX", "endY"],
            "optional": ["pid"],
            "types": {"pid": int, "startX": (int, float), "startY": (int, float), "endX": (int, float), "endY": (int, float)},
        },
        "window_mgmt": {
            "required": ["action"],
            "optional": ["pid", "x", "y", "width", "height"],
            "types": {"pid": int, "action": str, "x": (int, float), "y": (int, float), "width": (int, float), "height": (int, float)},
        },
        "macos-use_set_clipboard": {
            "required": ["text"],
            "optional": ["showAnimation", "animationDuration"],
            "types": {"text": str, "showAnimation": bool, "animationDuration": (int, float)},
        },
        "macos-use_get_clipboard": {
            "required": [],
            "types": {},
        },
        "macos-use_system_control": {
            "required": ["action"],
            "types": {"action": str},
        },
        "set_clipboard": {
            "required": ["text"],
            "optional": ["showAnimation", "animationDuration"],
            "types": {"text": str, "showAnimation": bool, "animationDuration": (int, float)},
        },
        "get_clipboard": {
            "required": [],
            "types": {},
        },
        "system_control": {
            "required": ["action"],
            "types": {"action": str},
        },
        "vibe_prompt": {
            "required": ["prompt"],
            "optional": ["cwd", "timeout_s", "output_format", "auto_approve", "max_turns"],
            "types": {"prompt": str, "cwd": str, "timeout_s": (int, float), "output_format": str, "auto_approve": bool, "max_turns": int},
        },
        "vibe_analyze_error": {
            "required": ["error_message"],
            "optional": ["log_context", "file_path", "cwd", "timeout_s", "auto_fix"],
            "types": {"error_message": str, "log_context": str, "file_path": str, "cwd": str, "timeout_s": (int, float), "auto_fix": bool},
        },
        "vibe_code_review": {
            "required": ["file_path"],
            "optional": ["focus_areas", "cwd", "timeout_s"],
            "types": {"file_path": str, "focus_areas": str, "cwd": str, "timeout_s": (int, float)},
        },
        "vibe_smart_plan": {
            "required": ["objective"],
            "optional": ["context", "cwd", "timeout_s"],
            "types": {"objective": str, "context": str, "cwd": str, "timeout_s": (int, float)},
        },
        "vibe_ask": {
            "required": ["question"],
            "optional": ["cwd", "timeout_s"],
            "types": {"question": str, "cwd": str, "timeout_s": (int, float)},
        },
        "restart_mcp_server": {
            "required": ["server_name"],
            "types": {"server_name": str},
        },
        "query_db": {
            "required": ["query"],
            "optional": ["params"],
            "types": {"query": str, "params": dict},
        },
    }

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

        # Try notes first (faster)
        try:
            result = await mcp_manager.call_tool(
                "notes",
                "search_notes",
                {
                    "category": "verification_report",
                    "tags": [f"step_{step_id}"],
                    "limit": 1,
                },
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
                notes = notes_result["notes"]
                if notes and len(notes) > 0:
                    note_id = notes[0]["id"]
                    note_result = await mcp_manager.call_tool(
                        "notes", "read_note", {"note_id": note_id}
                    )

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

        # --- SPECIAL CASE: Consent/Approval Steps ---
        # If the step is about getting user consent/approval/confirmation,
        # we can't automate it fully. Open Terminal and inform the user.
        step_action_lower = str(step.get("action", "")).lower()
        
        # FIX: More strict consent detection to prevent loops
        # Only treat as consent request if explicitly asked for user interaction
        is_consent_request = (
            "ask user" in step_action_lower
            or "request user consent" in step_action_lower
            or "await user approval" in step_action_lower
            or "get user confirmation" in step_action_lower
            or step.get("requires_consent", False) is True
        )

        if is_consent_request:
            logger.info(
                "[TETYANA] Detected consent/approval request step. Opening Terminal for user input."
            )
            # Open Terminal so user can see/respond
            import subprocess  # noqa: E402

            subprocess.run(["open", "-a", "Terminal"], check=False)

            # Create a simple message for the user
            consent_msg = """\nATLAS TRINITY SYSTEM REQUEST:

Action: {step.get('action')}
Expected: {step.get('expected_result', 'User confirmation')}

Please type your response below and press Enter:
(Type 'APPROVED' to proceed, 'REJECTED' to cancel)

> """

            # Since we can't wait for user input synchronously here,
            # we'll just mark this as successful and let the system continue.
            # The orchestrator will handle the actual consent logic.
            return StepResult(
                step_id=step.get("id", self.current_step),
                success=True,
                result="Terminal opened for user consent. Awaiting user response.",
                voice_message="Відкриваю термінал для запиту згоди. Будь ласка, підтвердіть.",
                error=None,
                tool_call={"name": "open_terminal", "args": {"message": consent_msg}},
            )

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

        # Fetch Grisha's feedback for retry attempts
        grisha_feedback = ""
        if attempt > 1:
            logger.info(f"[TETYANA] Attempt {attempt} - fetching Grisha's rejection feedback...")
            grisha_feedback = await self.get_grisha_feedback(step.get("id")) or ""

        target_server = step.get("realm") or step.get("tool") or step.get("server")
        # Normalize generic 'browser' realm to macos-use to leverage native automation
        if target_server == "browser":
            target_server = "macos-use"
        tools_summary = ""
        monologue = {}

        # SMART GATE: Check if we can skip reasoning
        # Only skip if it's the first attempt; on retries (attempt > 1), always use reasoning to address feedback
        skip_reasoning = (
            attempt == 1
            and target_server in SKIP_REASONING_TOOLS
            and step.get("tool")
            and step.get("args")
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
                global_goal=step.get("global_goal", ""),
                parent_action=step.get("parent_action", ""),
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
                        if "vibe" in action_text:
                            inferred_name = "vibe_prompt"
                        elif any(kw in action_text for kw in ["click", "type", "press", "scroll", "open app"]):
                            inferred_name = "macos-use"
                        elif any(kw in action_text for kw in ["file", "read", "write", "create", "save"]):
                            inferred_name = "filesystem"
                        elif any(kw in action_text for kw in ["run", "execute", "command", "terminal", "bash"]):
                            inferred_name = "terminal"
                    
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
                 v_res = await self._call_mcp_direct("vibe", "vibe_analyze_error", {
                     "error_message": error_msg,
                     "auto_fix": True
                 })
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
        """Executes the tool call with robust mapping and synonym support"""
        from ..logger import logger  # noqa: E402
        logger.info(f"[TETYANA] Dispatching tool: {tool_call.get('name', 'None')} with args: {str(tool_call.get('args', {}))[:200]}")

        # Safety check: ensure tool_call is a dict
        if not isinstance(tool_call, dict):
            logger.error(
                f"[TETYANA] Invalid tool_call type: {type(tool_call)}. Expected dict. Content: {tool_call}"
            )
            return {"success": False, "error": f"Invalid tool_call: {tool_call}"}

        # Robust name extraction
        tool_name = (tool_call.get("name") or tool_call.get("tool") or "").strip().lower()
        args = tool_call.get("args") or tool_call.get("arguments") or {}

        # --- INFER TOOL NAME IF MISSING ---
        if not tool_name:
            action_raw = str(args.get("action", "")).lower()
            command_raw = str(args.get("command", "")).lower()
            command_arg = args.get("command")
            logger.info(f"[TETYANA] Inferring tool. Action: '{action_raw[:50]}...', Command: '{command_raw[:50]}...'")
            
            # 7. Vibe - Ensure correct tool name and args
            if "vibe" in action_raw or "vibe" in command_raw:
                tool_name = "vibe_prompt"
                
                # Consolidate prompt
                prompt_text = args.get("prompt") or args.get("task") or args.get("instruction") or action_raw
                
                # Extract prompt from command list if present
                # Example: ['vibe_prompt', '-p', 'THE PROMPT', '--output', ...]
                if not prompt_text and isinstance(command_arg, list):
                    try:
                        if "-p" in command_arg:
                            idx = command_arg.index("-p")
                            if idx + 1 < len(command_arg):
                                prompt_text = command_arg[idx + 1]
                        elif "--prompt" in command_arg:
                             idx = command_arg.index("--prompt")
                             if idx + 1 < len(command_arg):
                                prompt_text = command_arg[idx + 1]
                    except Exception:
                        pass
                
                # If still empty, use the raw command string as fallback
                if not prompt_text:
                    prompt_text = str(command_arg)

                # Use valid args for vibe_prompt tool
                # output_format should be 'text' or 'json', NOT 'markdown' (which caused CLI crash)
                new_args = {
                    "prompt": prompt_text,
                    "output_format": "text", 
                    "timeout_s": 600  # Explicitly allow long timeout
                }
                
                logger.info(f"[TETYANA] Inferred tool: vibe_prompt with args: {new_args}")
                return await self._call_mcp_direct("vibe", "vibe_prompt", new_args)
            elif any(kw in action_raw for kw in ["click", "type", "press", "screenshot"]):
                tool_name = "macos-use"
                logger.info("[TETYANA] Inferred tool: macos-use")
            elif any(kw in action_raw for kw in ["read", "write", "list", "save"]):
                tool_name = "filesystem"
                logger.info("[TETYANA] Inferred tool: filesystem")
            elif action_raw:
                # If we have an action but no tool, try to use the action as the tool name
                tool_name = action_raw
                logger.info(f"[TETYANA] Using action as tool: {tool_name}")

        # --- INTERNAL SYSTEM TOOLS ---
        if tool_name in ["restart_mcp_server", "restart_server", "reboot_mcp"]:
            from ..mcp_manager import mcp_manager
            target = args.get("server_name") or args.get("name")
            if target:
                success = await mcp_manager.restart_server(target)
                return {
                    "success": success,
                    "output": f"MCP Server '{target}' has been restarted.",
                    "voice_message": f"Перезавантажую сервер {target}."
                }
            return {"success": False, "error": "Missing 'server_name' argument."}

        if tool_name in ["query_db", "database_query", "sql_query"]:
             from ..mcp_manager import mcp_manager
             query = args.get("query")
             if query:
                 results = await mcp_manager.query_db(query, args.get("params"))
                 return {
                     "success": True,
                     "output": f"Query Results: {json.dumps(results, indent=2)}",
                     "voice_message": "Виконую запит до бази даних."
                 }
             return {"success": False, "error": "Missing 'query' argument."}

        # --- UNIVERSAL TOOL SYNONYMS & INTENT ---
        terminal_synonyms = [
            "terminal",
            "bash",
            "zsh",
            "sh",
            "python",
            "python3",
            "pip",
            "pip3",
            "cmd",
            "run",
            "execute",
            "execute_command",
            "terminal_execute",
            "execute_terminal",
            "terminal.execute",
            "osascript",
            "applescript",
            "system",
            "os.system",
            "subprocess",
            "термінал",
            "curl",
            "wget",
            "jq",
            "grep",
            "git",
            "docker",
            "npm",
            "npx",
            "brew",
        ]
        fs_synonyms = ["filesystem", "fs", "file", "файлова система", "files"]
        scraper_synonyms = [
            "scrape_and_extract",
            "scrape",
            "extract",
            "scraper",
            "web_scraper",
            "justwatch_api",
            "reelgood_api",
        ]
        search_synonyms = ["duckduckgo_search", "duckduckgo", "search", "google", "bing", "ddg"]
        discovery_synonyms = ["list_tools", "help", "show_tools", "inspect"]

        # Intent-based routing
        if any(tool_name == syn or tool_name.split(".")[0] == syn for syn in terminal_synonyms):
            original_tool = tool_name
            tool_name = "terminal"
            # If the tool name was a command synonym, ensure it's in the args
            if original_tool in [
                "python",
                "python3",
                "pip",
                "pip3",
                "curl",
                "wget",
                "jq",
                "grep",
                "git",
                "docker",
                "npm",
                "npx",
                "brew",
            ]:
                cmd = args.get("command") or args.get("cmd") or args.get("args") or ""
                # Handle cases where args might be a list or dict
                if isinstance(cmd, (list, dict)):
                    cmd = str(cmd)
                args["command"] = f"{original_tool} {cmd}".strip()
            elif original_tool in ["osascript", "applescript"]:
                cmd = args.get("command") or args.get("cmd") or args.get("script") or ""
                args["command"] = (
                    f"osascript -e '{cmd}'" if "'" not in cmd else f'osascript -e "{cmd}"'
                )
            else:
                # Fallback for generic 'terminal' tool call matching
                cmd = (
                    args.get("command")
                    or args.get("cmd")
                    or args.get("code")
                    or args.get("script")
                    or args.get("args")
                    or args.get("action")
                )
                # Last resort: if only one arg and it's a string, assume it's the command
                if not cmd and args and len(args) == 1:
                    first_val = list(args.values())[0]
                    if isinstance(first_val, str):
                        cmd = first_val

                if cmd:
                    if isinstance(cmd, (list, dict)):
                        cmd = str(cmd)
                    args["command"] = str(cmd)
            logger.info(f"[TETYANA] Syn-map: {original_tool} -> {tool_name}")

        elif any(tool_name == syn or tool_name.split(".")[0] == syn for syn in fs_synonyms):
            tool_name = "filesystem"
            logger.info("[TETYANA] FS-map: routing to filesystem")

        elif any(tool_name == syn or tool_name.split(".")[0] == syn for syn in scraper_synonyms):
            # Map 'scrape_and_extract' or 'justwatch_api' to 'fetch' if URL provided, else search
            if "url" in args or "urls" in args:
                tool_name = "fetch"
                logger.info(f"[TETYANA] Scraper-map: {tool_name} -> fetch")
            else:
                tool_name = "duckduckgo-search"
                logger.info(f"[TETYANA] Scraper-map: {tool_name} -> duckduckgo-search")

        elif any(tool_name == syn or tool_name.split(".")[0] == syn for syn in search_synonyms):
            tool_name = "duckduckgo-search"
            logger.info(f"[TETYANA] Search-map: {tool_name} -> duckduckgo-search")

        elif any(tool_name == syn for syn in discovery_synonyms):
            # Meta-tool call: list_tools
            logger.info("[TETYANA] Meta-map: list_tools -> dynamic catalog")
            catalog = await mcp_manager.get_mcp_catalog()
            return {"success": True, "output": f"Available Realms and basic tools:\n{catalog}"}

        # Detect mkdir/cat direct hallucinations
        if tool_name in ["mkdir", "ls", "cat", "rm", "mv", "cp", "touch"]:
            action = tool_name
            tool_name = "terminal"
            cmd = args.get("command") or args.get("cmd") or args.get("path") or ""
            args["command"] = f"{action} {cmd}".strip()
            logger.info(f"[TETYANA] CMD-map: {action} -> terminal")

        # Handle "server.tool" dots
        if "." in tool_name:
            parts = tool_name.split(".")
            server = parts[0]
            action = parts[1]
            # Map common hallucinant formats
            if server == "filesystem":
                tool_name = "filesystem"
                args["action"] = action
                # Compat: list_dir -> list_directory
                if action == "list_dir":
                    args["action"] = "list_directory"
            elif server == "terminal":
                tool_name = "terminal"
                args["command"] = args.get("command") or args.get("cmd") or action
            elif server == "macos-use":
                # macos-use.tool_name -> dispatch to macos-use server
                tool_name = action
                tool_call["server"] = "macos-use"
                logger.info(f"[TETYANA] macOS-use Dot-map: macos-use.{action}")
            logger.info(f"[TETYANA] Dot-map: {server}.{action} -> {tool_name}")

        # --- MACOS-USE TOOL MAPPING ---
        # Handle hallucinated tool names like 'macos-use_open_application_and_traverse'
        # These should be dispatched to macos-use server with proper tool name
        macos_use_tools = [
            "macos-use_open_application_and_traverse",
            "macos-use_click_and_traverse",
            "macos-use_type_and_traverse",
            "macos-use_press_key_and_traverse",
            "macos-use_refresh_traversal",
        ]
        if tool_name in macos_use_tools or any(
            tool_name.startswith(prefix) for prefix in ["macos-use_", "macos_use_"]
        ):
            # Extract actual tool name and set server
            actual_tool = tool_name
            if tool_name.startswith("macos-use_"):
                actual_tool = tool_name  # Keep as is, it's the correct format
            elif tool_name.startswith("macos_use_"):
                actual_tool = tool_name.replace("macos_use_", "macos-use_")
            tool_call["server"] = "macos-use"
            tool_name = actual_tool
            logger.info(f"[TETYANA] macOS-use map: {tool_name} -> macos-use.{actual_tool}")

        # PATH EXPANSION using SharedContext
        if "path" in args and isinstance(args["path"], str):
            original = args["path"]
            args["path"] = shared_context.resolve_path(original)
            if args["path"] != original:
                logger.info(f"[TETYANA] Path resolved: {original} -> {args['path']}")

        # Patches for hallucinated plural arguments
        if tool_name == "fetch" or tool_call.get("server") == "fetch":
            if "urls" in args and "url" not in args:
                urls = args["urls"]
                if isinstance(urls, list) and len(urls) > 0:
                    args["url"] = urls[0]  # Take first for now
                    logger.info(f"[TETYANA] Argument patch: urls -> url (using {args['url']})")

        # Direct MCP call
        if tool_name == "mcp":
            server = tool_call.get("server") or args.get("server")
            mcp_tool = tool_call.get("tool_name") or args.get("tool_name")
            mcp_args = tool_call.get("args") or args.get("args") or args
            return await self._call_mcp_direct(server, mcp_tool, mcp_args)

        # --- DYNAMIC MCP DISPATCHER ---
        from ..mcp_manager import mcp_manager  # noqa: E402

        explicit_server = tool_call.get("server")
        # Normalize common aliases
        if explicit_server == "browser":
            explicit_server = "macos-use"
            tool_call["server"] = explicit_server

        if explicit_server:
            # Handle realm-as-tool name in explicit dispatch
            if tool_name == explicit_server:
                default_map = {
                    "fetch": "fetch",
                    "duckduckgo-search": "duckduckgo_search",
                    "terminal": "execute_command",
                    "macos-use": "screenshot",
                    "time": "get_current_time",
                    "sequential-thinking": "sequentialthinking",
                    "filesystem": "read_file",
                    "vibe": "vibe_prompt",
                }
                if explicit_server in default_map:
                    tool_name = default_map[explicit_server]

            logger.info(f"[TETYANA] Explicit server dispatch: {explicit_server}.{tool_name}")
            # Execute the direct MCP call first
            res = await self._call_mcp_direct(explicit_server, tool_name, args)

            # If this was a puppeteer/browser call that navigated or clicked, try to collect
            # verification artifacts (title, HTML, screenshot) so Grisha can verify the step.
            try:
                if explicit_server in ("macos-use", "browser"):
                    # Check if action was a browser-like navigation
                    if (
                        tool_name in ("macos-use_open_application_and_traverse")
                        or tool_name.endswith("navigate")
                    ):
                        # small delay to allow rendering
                        await asyncio.sleep(1.5)
                        from ..logger import logger  # noqa: E402
                        logger.info(
                            f"[TETYANA] (explicit dispatch) Collecting artifacts for step {args.get('step_id')}"
                        )
                        # We use native screenshot for verification instead of puppeteer
                        shot_path = await self._take_screenshot_for_vision()
                        if shot_path:
                            logger.info(f"[TETYANA] Saved screenshot artifact via macos-use: {shot_path}")

                        # Save artifacts (inline to avoid refactor)
                        try:
                            import base64  # noqa: E402

                            from ..config import SCREENSHOTS_DIR, WORKSPACE_DIR  # noqa: E402
                            from ..mcp_manager import mcp_manager  # noqa: E402

                            ts = __import__("time").strftime("%Y%m%d_%H%M%S")
                            artifacts = []

                            if html_text:
                                html_file = (
                                    WORKSPACE_DIR / f"grisha_step_{args.get('step_id')}_{ts}.html"
                                )
                                html_file.parent.mkdir(parents=True, exist_ok=True)
                                html_file.write_text(html_text, encoding="utf-8")
                                artifacts.append(str(html_file))
                                logger.info(f"[TETYANA] Saved HTML artifact: {html_file}")

                            if screenshot_b64:
                                SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                                img_file = (
                                    SCREENSHOTS_DIR / f"grisha_step_{args.get('step_id')}_{ts}.png"
                                )
                                with open(img_file, "wb") as f:
                                    f.write(base64.b64decode(screenshot_b64))
                                artifacts.append(str(img_file))
                                logger.info(f"[TETYANA] Saved screenshot artifact: {img_file}")

                            # Prepare note content
                            note_title = f"Grisha Artifact - Step {args.get('step_id')} @ {ts}"
                            snippet = ""
                            if title_text:
                                snippet += f"Title: {title_text}\n\n"
                            if html_text:
                                snippet += f"HTML Snippet:\n{(html_text[:1000] + '...') if len(html_text) > 1000 else html_text}\n\n"

                            detected = []
                            if html_text:
                                keywords = ["phone", "sms", "verification", "код", "телефон"]
                                low = html_text.lower()
                                for kw in keywords:
                                    if kw in low:
                                        detected.append(kw)

                            note_content = (
                                f"Artifacts for step {args.get('step_id')} saved at {ts}.\n\nFiles:\n"
                                + (
                                    "\n".join(artifacts)
                                    if artifacts
                                    else "(no binary files captured)"
                                )
                                + "\n\n"
                                + snippet
                            )
                            if detected:
                                note_content += (
                                    f"Detected keywords in HTML: {', '.join(detected)}\n"
                                )

                            try:
                                await mcp_manager.call_tool(
                                    "notes",
                                    "create_note",
                                    {
                                        "title": note_title,
                                        "content": note_content,
                                        "category": "verification_artifact",
                                        "tags": ["grisha", f"step_{args.get('step_id')}"],
                                    },
                                )
                                logger.info(
                                    f"[TETYANA] Created verification artifact note for step {args.get('step_id')}"
                                )
                            except Exception as e:
                                logger.warning(f"[TETYANA] Failed to create artifact note: {e}")

                            try:
                                await mcp_manager.call_tool(
                                    "memory",
                                    "create_entities",
                                    {
                                        "entities": [
                                            {
                                                "name": f"grisha_artifact_step_{args.get('step_id')}",
                                                "entityType": "artifact",
                                                "observations": artifacts,
                                            }
                                        ]
                                    },
                                )
                                logger.info(
                                    f"[TETYANA] Created memory artifact for step {args.get('step_id')}"
                                )
                            except Exception as e:
                                logger.warning(f"[TETYANA] Failed to create memory artifact: {e}")
                        except Exception as e:
                            logger.warning(
                                f"[TETYANA] explicit dispatch _save_artifacts exception: {e}"
                            )
            except Exception as e:
                from ..logger import logger  # noqa: E402

                logger.warning(f"[TETYANA] Explicit dispatch artifact collection failed: {e}")

            return res

        servers = mcp_manager.config.get("mcpServers", {})
        if tool_name in servers and not tool_name.startswith("_"):
            server_name = tool_name
            logger.info(f"[TETYANA] Server-map: detected direct server call to '{server_name}'")

            # Use action as tool name if present, else fallback to common ones
            mcp_tool = args.get("action") or args.get("tool") or server_name

            # --- PATCH: Manual Mappings for MCP Servers ---
            # Filesystem Fixes
            if server_name == "filesystem":
                if mcp_tool in ["list_dir", "ls", "ls -l"]:
                    mcp_tool = "list_directory"
                if mcp_tool in ["mkdir", "create_dir", "makedir"]:
                    mcp_tool = "create_directory"
                if mcp_tool in ["write", "save"]:
                    mcp_tool = "write_file"
                if mcp_tool in ["read", "cat"]:
                    mcp_tool = "read_file"
                if mcp_tool in ["is_dir", "exists", "stat", "get_info"]:
                    mcp_tool = "get_file_info"

            # MacOS Use Fixes
            if server_name == "macos-use":
                # Robust mappings for Swift binary tools
                if mcp_tool in ["click", "click_and_traverse"]:
                    mcp_tool = "macos-use_click_and_traverse"
                if mcp_tool in ["type", "type_and_traverse", "write"]:
                    mcp_tool = "macos-use_type_and_traverse"
                if mcp_tool in ["press", "press_key", "hotkey", "key", "press_key_and_traverse"]:
                    mcp_tool = "macos-use_press_key_and_traverse"
                if mcp_tool in ["refresh", "tree", "get_tree", "refresh_traversal"]:
                    mcp_tool = "macos-use_refresh_traversal"
                if mcp_tool in ["open", "launch", "open_app"]:
                    mcp_tool = "macos-use_open_application_and_traverse"
                    # Map common args to 'identifier'
                    if "identifier" not in args:
                        args["identifier"] = (
                            args.get("app_name")
                            or args.get("path")
                            or args.get("name")
                            or args.get("app")
                            or ""
                        )

                if mcp_tool == "screenshot":
                    mcp_tool = "macos-use_take_screenshot"
                
                if mcp_tool in ["analyze", "vision", "ocr", "analyze_screen"]:
                    mcp_tool = "macos-use_analyze_screen"

                if mcp_tool in ["right_click", "right_click_and_traverse"]:
                    mcp_tool = "macos-use_right_click_and_traverse"
                
                if mcp_tool in ["double_click", "double_click_and_traverse"]:
                    mcp_tool = "macos-use_double_click_and_traverse"

                if mcp_tool in ["drag_drop", "drag_and_drop", "drag_and_drop_and_traverse"]:
                    mcp_tool = "macos-use_drag_and_drop_and_traverse"

                if mcp_tool in ["scroll", "scroll_and_traverse"]:
                    mcp_tool = "macos-use_scroll_and_traverse"

                if mcp_tool in ["window_management", "window_mgmt"]:
                    mcp_tool = "macos-use_window_management"

                # No Vibe fixes here anymore, moved to _call_mcp_direct

                if mcp_tool in ["set_clipboard", "clipboard_set", "copy"]:
                    mcp_tool = "macos-use_set_clipboard"

                if mcp_tool in ["get_clipboard", "clipboard_get", "paste"]:
                    mcp_tool = "macos-use_get_clipboard"

                if mcp_tool in ["system_control", "media_control", "volume", "brightness"]:
                    mcp_tool = "macos-use_system_control"

            # Common server-to-default-tool mappings
            default_map = {
                "fetch": "fetch_url",
                "duckduckgo-search": "search",
                "terminal": "execute_command",
                "macos-use": "screenshot",  # Default to screenshot if just 'macos-use' (handled above if mapped)
                "time": "get_current_time",
                "sequential-thinking": "sequentialthinking",
                "filesystem": "read_file",
            }
            if mcp_tool == server_name and server_name in default_map:
                mcp_tool = default_map[server_name]

            return await self._call_mcp_direct(server_name, mcp_tool, args)

        # Wrappers
        if tool_name == "terminal":
            # Map legacy 'terminal' wrapper directly to 'execute_command'
            # This handles cases where LLM still tries to use 'terminal' tool name
            return await self._run_terminal_command(args)
        elif tool_name == "gui":
            return await self._perform_gui_action(args)
        elif tool_name == "browser":
            return await self._browser_action(args)
        # 2. Terminal - Normalize arguments
        elif tool_name == "execute_command" or tool_name == "run_command":
            if "actions" in args and isinstance(args["actions"], list):
                 # Extract the first terminal command from actions list
                 for action in args["actions"]:
                     if action.get("type") == "terminal" or action.get("command"):
                         args["command"] = action.get("command")
                         break
            
            if "command" not in args and "cmd" in args:
                args["command"] = args.pop("cmd")
            return await self._run_terminal_command(args)
        elif tool_name == "filesystem" or tool_name == "editor":
            if tool_name == "editor" and "action" not in args:
                if "content" in args:
                    args["action"] = "write_file"
                elif "path" in args:
                    args["action"] = "read_file"
            return await self._filesystem_action(args)
        elif tool_name == "applescript":
            return await self._applescript_action(args)
        elif tool_name == "search":
            return await self._search_action(args)
        elif tool_name == "github":
            return await self._github_action(args)
        else:
            # Unknown tool - provide helpful error message
            error_msg = f"Unknown tool: {tool_name}. This tool is not available in the current MCP configuration."
            logger.warning(f"[TETYANA] {error_msg}")
            logger.warning(f"[TETYANA] Tool call details: {tool_call}")

            # For consent/approval related tools, provide guidance
            consent_tool_names = [
                "generate_consent",
                "generate_structured_text",
                "assistant",
                "text_generation",
            ]
            if any(consent in tool_name for consent in consent_tool_names):
                error_msg += " For consent/approval steps, consider using Terminal to communicate with the user directly."

            return {"success": False, "error": error_msg}

    def _validate_macos_use_args(self, tool_name: str, args: Dict) -> Dict:
        """Validate and normalize arguments for macos-use Swift binary tools"""
        from ..logger import logger  # noqa: E402

        if tool_name not in self.MACOS_USE_SCHEMAS:
            return args

        schema = self.MACOS_USE_SCHEMAS[tool_name]
        validated = {}

        # Check required args
        for arg_name in schema.get("required", []):
            if arg_name not in args:
                logger.warning(f"[TETYANA] Missing required argument '{arg_name}' for {tool_name}")
                # Try common aliases
                aliases = {
                    "identifier": ["app", "app_name", "application", "name"],
                    "pid": ["process_id", "processId", "PID"],
                    "text": ["content", "value", "string"],
                    "keyName": ["key", "keyname", "key_name"],
                    "x": ["X", "xCoord", "x_coord"],
                    "y": ["Y", "yCoord", "y_coord"],
                }
                for alias in aliases.get(arg_name, []):
                    if alias in args:
                        validated[arg_name] = args[alias]
                        logger.info(f"[TETYANA] Alias found: {alias} -> {arg_name}")
                        break
                else:
                    raise ValueError(f"Missing required argument '{arg_name}' for {tool_name}")
            else:
                validated[arg_name] = args[arg_name]

        # Copy optional args
        for arg_name in schema.get("optional", []):
            if arg_name in args:
                validated[arg_name] = args[arg_name]

        # Type coercion for pid (must be int)
        if "pid" in validated and not isinstance(validated["pid"], int):
            try:
                validated["pid"] = int(validated["pid"])
                logger.info(f"[TETYANA] Type coercion: pid -> int ({validated['pid']})")
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid pid value: {validated['pid']}") from e

        # Type coercion for coordinates (must be float)
        for coord in ["x", "y", "animationDuration"]:
            if coord in validated:
                try:
                    validated[coord] = float(validated[coord])
                    logger.info(f"[TETYANA] Type coercion: {coord} -> float ({validated[coord]})")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {coord} value: {validated[coord]}") from e

        # Type coercion for showAnimation (must be bool)
        if "showAnimation" in validated:
            if isinstance(validated["showAnimation"], str):
                validated["showAnimation"] = validated["showAnimation"].lower() == "true"
            else:
                validated["showAnimation"] = bool(validated["showAnimation"])

        # Ensure modifierFlags is a list of strings
        if "modifierFlags" in validated:
            flags = validated["modifierFlags"]
            if isinstance(flags, str):
                validated["modifierFlags"] = [flags]
            elif not isinstance(flags, list):
                validated["modifierFlags"] = []

        return validated

    async def _call_mcp_direct(self, server: str, tool: str, args: Dict) -> Dict[str, Any]:
        from ..logger import logger  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        try:
            # MACOS-USE VALIDATION & MAPPING
            if server == "macos-use":
                # Map legacy/hallucinated tool names to official ones
                tool_map = {
                    "terminal": "execute_command",
                    "run_command": "execute_command",
                    "exec": "execute_command",
                    "command": "execute_command",
                    "screenshot": "macos-use_take_screenshot",
                    "take_screenshot": "macos-use_take_screenshot",
                    "capture": "macos-use_take_screenshot",
                    "vision": "macos-use_analyze_screen",
                    "analyze": "macos-use_analyze_screen",
                    "analyze_screen": "macos-use_analyze_screen",
                    "ocr": "macos-use_analyze_screen",
                    "scan": "macos-use_analyze_screen",
                    "scroll": "macos-use_scroll_and_traverse",
                    "right_click": "macos-use_right_click_and_traverse",
                    "double_click": "macos-use_double_click_and_traverse",
                    "drag_drop": "macos-use_drag_and_drop_and_traverse",
                    "window_mgmt": "macos-use_window_management",
                    "minimize": "macos-use_window_management",
                    "maximize": "macos-use_window_management",
                    "resize": "macos-use_window_management",
                    "move": "macos-use_window_management",
                    "set_clipboard": "macos-use_set_clipboard",
                    "get_clipboard": "macos-use_get_clipboard",
                    "clipboard": "macos-use_get_clipboard",
                    "system_control": "macos-use_system_control",
                    "media": "macos-use_system_control",
                    "volume": "macos-use_system_control",
                }
                if tool in tool_map:
                    logger.info(f"[TETYANA] Auto-mapping tool '{tool}' -> '{tool_map[tool]}'")
                    tool = tool_map[tool]

            # Server Name Fixes
            if server == "sequentialthinking":
                server = "sequential-thinking"
            
            if server == "sequential-thinking":
                 # Ensure tool name matches what the server expects
                 if tool == "sequential-thinking":
                     tool = "sequentialthinking"

            # VIBE VALIDATION & MAPPING
            if server == "vibe":
                # If tool is generic 'vibe' or missing, try to infer from args
                if tool == "vibe":
                    if "objective" in args or "plan" in args:
                        tool = "vibe_smart_plan"
                    elif "question" in args:
                        tool = "vibe_ask"
                    elif "error_message" in args:
                        tool = "vibe_analyze_error"
                    else:
                        tool = "vibe_prompt"

                # Map common aliases/hallucinations for tool names
                vibe_tools = {
                    "vibe": "vibe_prompt",
                    "prompt": "vibe_prompt",
                    "chat": "vibe_prompt",
                    "ask_vibe": "vibe_prompt",
                    "smart_plan": "vibe_smart_plan",
                    "design": "vibe_smart_plan",
                    "plan": "vibe_smart_plan",
                    "analyze_error": "vibe_analyze_error",
                    "debug": "vibe_analyze_error",
                    "self-healing": "vibe_analyze_error",
                    "fix": "vibe_analyze_error",
                    "code_review": "vibe_code_review",
                    "review": "vibe_code_review",
                    "ask": "vibe_ask",
                    "question": "vibe_ask"
                }
                if tool in vibe_tools:
                    tool = vibe_tools[tool]

                # Normalize arguments based on tool name
                if tool == "vibe_prompt":
                    if "objective" in args and "prompt" not in args:
                        args["prompt"] = args.pop("objective")
                    if "question" in args and "prompt" not in args:
                        args["prompt"] = args.pop("question")
                elif tool == "vibe_smart_plan":
                    if "prompt" in args and "objective" not in args:
                        args["objective"] = args.pop("prompt")
                    if "question" in args and "objective" not in args:
                        args["objective"] = args.pop("question")
                elif tool == "vibe_ask":
                    if "prompt" in args and "question" not in args:
                        args["question"] = args.pop("prompt")
                    if "objective" in args and "question" not in args:
                        args["question"] = args.pop("objective")
                elif tool == "vibe_analyze_error":
                    if "prompt" in args and "error_message" not in args:
                        args["error_message"] = args.pop("prompt")

                logger.info(f"[TETYANA] Vibe-map: final tool={tool}, args={list(args.keys())}")

                # ENFORCE DYNAMIC TIMEOUT (User request: Don't let it loop at 300s)
                # If agent provides a timeout smaller than config, override it.
                vibe_config = config.get("vibe", {})
                config_timeout = vibe_config.get("timeout_s", 600) if isinstance(vibe_config, dict) else 600
                
                if args.get("timeout_s", 0) < config_timeout:
                    logger.info(f"[TETYANA] Auto-extending timeout_s from {args.get('timeout_s')} to {config_timeout} (based on global config)")
                    args["timeout_s"] = config_timeout

                try:
                    args = self._validate_macos_use_args(tool, args)
                    logger.info(f"[TETYANA] Validated macos-use args: {args}")
                except ValueError as ve:
                    return {"success": False, "error": f"Argument validation failed: {ve}"}

            res = await mcp_manager.call_tool(server, tool, args)
            return self._format_mcp_result(res)
        except Exception as e:
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
                    await mcp_manager.call_tool(
                        "notes",
                        "create_note",
                        {
                            "title": note_title,
                            "content": note_content,
                            "category": "verification_artifact",
                            "tags": ["grisha", f"step_{step_id}"],
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
                "error": f"Unknown FS action: {action}. Valid: read_file, write_file, create_directory, list_directory",
            }

    async def _search_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Web search via Brave MCP"""
        from ..mcp_manager import mcp_manager  # noqa: E402

        query = args.get("query", "")
        # Tool name usually 'duckduckgo_search' or just 'search'
        res = await mcp_manager.call_tool(
            "duckduckgo-search", "duckduckgo_search", {"query": query}
        )
        return self._format_mcp_result(res)

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

        # Translate commonly used technical prefixes if English
        essence = essence.lower()
        if essence.startswith("create"):
            essence = essence.replace("create", "Створюю", 1)
        elif essence.startswith("update"):
            essence = essence.replace("update", "Оновлюю", 1)
        elif essence.startswith("check"):
            essence = essence.replace("check", "Перевіряю", 1)
        elif essence.startswith("install"):
            essence = essence.replace("install", "Встановлюю", 1)
        elif essence.startswith("run"):
            essence = essence.replace("run", "Запускаю", 1)
        elif essence.startswith("execute"):
            essence = essence.replace("execute", "Виконую", 1)

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
        """Parse JSON response from LLM"""
        import json  # noqa: E402

        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        # ВИПРАВЛЕННЯ: обробка GitHub API timeout
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

        return {"raw": content}
