"""
Grisha - The Visor/Auditor

Role: Result verification via Vision, Security control
Voice: Mykyta (male)
Model: GPT-4o (Vision)
"""

import base64
import os
# Robust path handling for both Dev and Production (Packaged)
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

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
from ..logger import logger  # noqa: E402
from ..prompts import AgentPrompts  # noqa: E402


@dataclass
class VerificationResult:
    """Verification result"""

    step_id: int
    verified: bool
    confidence: float  # 0.0 - 1.0
    description: str
    issues: list
    voice_message: str = ""
    timestamp: datetime = None
    screenshot_analyzed: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class Grisha:
    """
    Grisha - The Visor/Auditor

    Functions:
    - Verifying execution results via Vision
    - Analyzing screenshots
    - Security control (blocking dangerous actions)
    - Confirming or rejecting steps
    """

    NAME = AgentPrompts.GRISHA["NAME"]
    DISPLAY_NAME = AgentPrompts.GRISHA["DISPLAY_NAME"]
    VOICE = AgentPrompts.GRISHA["VOICE"]
    COLOR = AgentPrompts.GRISHA["COLOR"]
    SYSTEM_PROMPT = AgentPrompts.GRISHA["SYSTEM_PROMPT"]

    # Hardcoded blocklist for critical commands
    BLOCKLIST = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        ":(){:|:&};:",
        "chmod 777 /",
        "chown root:root /",
        "> /dev/sda",
        "mv / /dev/null",
    ]

    def __init__(self, vision_model: str = "gpt-4o"):
        # Get model config (config.yaml > parameter > env variables)
        agent_config = config.get_agent_config("grisha")
        security_config = config.get_security_config()

        final_model = vision_model
        if vision_model == "gpt-4o":  # default parameter
            final_model = agent_config.get("vision_model") or os.getenv(
                "VISION_MODEL", "gpt-4o"
            )

        self.llm = CopilotLLM(model_name=final_model, vision_model_name=final_model)
        self.temperature = agent_config.get("temperature", 0.3)

        # Load dangerous commands from config with fallback to hardcoded BLOCKLIST
        self.dangerous_commands = security_config.get(
            "dangerous_commands", self.BLOCKLIST
        )
        self.verifications: list = []

        # OPTIMIZATION: Strategy cache to avoid redundant LLM calls
        self._strategy_cache = {}

        # Reasoner Model (Raptor-Mini) for Strategy Planning
        # Default to raptor-mini, or use from config/env
        strategy_model = agent_config.get("strategy_model") or os.getenv(
            "STRATEGY_MODEL", "raptor-mini"
        )
        self.strategist = CopilotLLM(model_name=strategy_model)
        logger.info(
            f"[GRISHA] Initialized with Vision={final_model}, Strategy={strategy_model}"
        )

    async def _plan_verification_strategy(
        self,
        step_action: str,
        expected_result: str,
        context: dict,
        overall_goal: str = "",
    ) -> str:
        """
        Uses Raptor-Mini (MSP Reasoning) to create a robust verification strategy.
        OPTIMIZATION: Caches strategies by step type to avoid redundant LLM calls.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # OPTIMIZATION: Check cache first
        cache_key = f"{step_action[:50]}:{expected_result[:50]}"
        if cache_key in self._strategy_cache:
            logger.info(f"[GRISHA] Using cached strategy for: {cache_key[:30]}...")
            return self._strategy_cache[cache_key]

        prompt = AgentPrompts.grisha_strategy_prompt(
            step_action, expected_result, context, overall_goal=overall_goal
        )

        # Decide which verification stack to prefer based on environment and step
        decision = self._decide_verification_stack(step_action, expected_result, context)
        decision_context = f"prefer_vision_first={decision['prefer_vision_first']}; use_mcp={decision['use_mcp']}; preferred_servers={decision['preferred_servers']}; rationale={decision['rationale']}"
        system_msg = AgentPrompts.grisha_strategist_system_prompt(decision_context)
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.strategist.ainvoke(messages)
            strategy = getattr(response, "content", str(response))
            logger.info(f"[GRISHA] Strategy devised: {strategy[:200]}...")
            # Cache the strategy
            self._strategy_cache[cache_key] = strategy
            return strategy
        except Exception as e:
            logger.warning(f"[GRISHA] Strategy planning failed: {e}")
            return "Proceed with standard verification (Vision + Tools)."

    def _check_blocklist(self, action_desc: str) -> bool:
        """Check if action contains blocked commands"""
        for blocked in self.dangerous_commands:
            if blocked in action_desc:
                return True
        return False

    def _decide_verification_stack(self, step_action: str, expected_result: str, context: dict) -> dict:
        """
        Decide whether to prefer Vision or MCP tools based on:
        - whether the step is visual (ui/screenshot/window)
        - whether the step needs authoritative system/data checks (files, install, permissions)
        - availability of powerful Vision model
        - presence of local Swift-based MCP servers (prefer for low-latency authoritative checks)

        Returns a dict: {prefer_vision_first, use_mcp, preferred_servers, rationale}
        """
        visual_keywords = ("visual", "screenshot", "ui", "interface", "window", "dialog", "button", "icon")
        system_keywords = (
            "file",
            "install",
            "chmod",
            "chown",
            "git",
            "permission",
            "plist",
            "service",
            "daemon",
            "launchctl",
            "package",
            "brew",
            "pip",
            "npm",
            "run",
            "execute",
            "terminal",
        )

        visual_needed = any(
            kw in expected_result.lower() or kw in step_action.lower()
            for kw in visual_keywords
        )
        system_needed = any(
            kw in expected_result.lower() or kw in step_action.lower()
            for kw in system_keywords
        )

        try:
            from ..mcp_manager import mcp_manager

            servers_cfg = getattr(mcp_manager, "config", {}).get("mcpServers", {})
            servers = list(servers_cfg.keys())
            swift_servers = []
            for s, cfg in servers_cfg.items():
                if "swift" in s.lower() or "macos" in s.lower() or "mcp-server-macos-use" in s.lower():
                    swift_servers.append(s)
                else:
                    cmd = (cfg or {}).get("command", "") or ""
                    if isinstance(cmd, str) and "swift" in cmd.lower():
                        swift_servers.append(s)
        except Exception:
            servers = []
            swift_servers = []

        vision_model_name = (getattr(self.llm, "model_name", "") or "").lower()
        vision_powerful = any(x in vision_model_name for x in ("gpt-4o", "vision"))

        # Heuristic decision rules
        prefer_vision_first = bool(visual_needed or (vision_powerful and not system_needed))
        use_mcp = bool(system_needed or swift_servers)

        preferred_servers = []
        if use_mcp:
            preferred_servers = swift_servers if swift_servers else servers

        rationale = (
            f"visual_needed={visual_needed}, system_needed={system_needed}, "
            f"vision_powerful={vision_powerful}, swift_servers={swift_servers}"
        )
        return {
            "prefer_vision_first": prefer_vision_first,
            "use_mcp": use_mcp,
            "preferred_servers": preferred_servers,
            "rationale": rationale,
        }

    @retry(
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _call_tool_with_retry(
        self, mcp_manager, server: str, tool: str, args: dict
    ):
        """
        Call MCP tool with exponential backoff retry.
        Handles transient connection issues gracefully.
        """
        return await mcp_manager.call_tool(server, tool, args)

    async def verify_step(
        self,
        step: Dict[str, Any],
        result: Dict[str, Any],
        screenshot_path: Optional[str] = None,
        overall_goal: str = "",
    ) -> VerificationResult:
        """
        Verifies the result of step execution using Vision and MCP Tools
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from ..mcp_manager import mcp_manager

        step_id = step.get("id", 0)
        expected = step.get("expected_result", "")

        # PRIORITY: Use MCP tools first, screenshots only when explicitly needed
        # Only take screenshot if explicitly requested or if visual verification is clearly needed
        visual_verification_needed = (
            "visual" in expected.lower() or 
            "screenshot" in expected.lower() or
            "ui" in expected.lower() or
            "interface" in expected.lower() or
            "window" in expected.lower()
        )

        if (
            not visual_verification_needed
            or not screenshot_path
            or not isinstance(screenshot_path, str)
            or not os.path.exists(screenshot_path)
        ):
            screenshot_path = None

        # If we don't already have a screenshot path, try to find artifacts saved by Tetyana
        if not screenshot_path:
            try:
                from ..mcp_manager import mcp_manager
                notes_search = await mcp_manager.call_tool(
                    "notes",
                    "search_notes",
                    {"tags": [f"step_{step_id}"], "limit": 5},
                )
                if isinstance(notes_search, dict) and notes_search.get("result", {}).get("success"):
                    for n in notes_search["result"].get("notes", []):
                        note_file = n.get("file")
                        if note_file and os.path.exists(note_file):
                            try:
                                text = open(note_file, "r", encoding="utf-8").read()
                                import re

                                m = re.search(r"(/[^\s']+?\.png)", text)
                                if m:
                                    candidate = m.group(1)
                                    if os.path.exists(candidate):
                                        screenshot_path = candidate
                                        logger.info(f"[GRISHA] Found screenshot from notes: {candidate}")
                                        break
                            except Exception:
                                pass
            except Exception as e:
                logger.warning(f"[GRISHA] Could not search notes for screenshot: {e}")

        context_info = shared_context.to_dict()

        if hasattr(result, "result") and not isinstance(result, dict):
            actual = result.result
        elif isinstance(result, dict):
            actual = result.get("result", result.get("output", ""))
        else:
            actual = str(result)

        # 1. PLAN STRATEGY with Raptor-Mini
        strategy_context = await self._plan_verification_strategy(
            step.get("action", ""), expected, context_info, overall_goal=overall_goal
        )

        verification_history = []
        max_attempts = 3  # OPTIMIZATION: Reduced from 5 for faster verification
        attach_screenshot_next = False

        for attempt in range(max_attempts):
            content = []

            prompt_text = AgentPrompts.grisha_verification_prompt(
                strategy_context,
                step_id,
                step.get("action", ""),
                expected,
                actual,
                context_info,
                verification_history,
                overall_goal=overall_goal,
            )
            content.append({"type": "text", "text": prompt_text})

            if (
                screenshot_path
                and os.path.exists(screenshot_path)
                and (attempt == 0 or attach_screenshot_next)
            ):
                with open(screenshot_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                mime = "image/png"
                sp = screenshot_path.lower()
                if sp.endswith(".jpg") or sp.endswith(".jpeg"):
                    mime = "image/jpeg"
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_data}"},
                    }
                )
                attach_screenshot_next = False

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=content),
            ]

            response = await self.llm.ainvoke(messages)
            data = self._parse_response(response.content)

            if data.get("action") == "call_tool":
                server = data.get("server")
                tool = data.get("tool")
                args = data.get("arguments", {})

                if "." in tool:
                    parts = tool.split(".")
                    tool = parts[-1]
                    logger.warning(
                        f"[GRISHA] Stripped prefix from tool: {data.get('tool')} -> {tool}"
                    )

                if server in ["local", "server", "default"]:
                    if tool in ["execute_command", "terminal", "run", "shell"]:
                        server = "terminal"
                        tool = "execute_command"
                    else:
                        server = "filesystem"
                    logger.warning(
                        f"[GRISHA] Fixed hallucinated server: {data.get('server')} -> {server}"
                    )

                if server == "terminal" and tool in [
                    "terminal",
                    "run",
                    "execute",
                    "shell",
                ]:
                    tool = "execute_command"
                if server == "filesystem":
                    if tool in ["list", "ls", "dir"]:
                        tool = "list_directory"
                    if tool in ["read", "cat", "get"]:
                        tool = "read_file"
                    if tool in ["exists", "check", "file_exists"]:
                        tool = "get_file_info"

                if server == "sequential-thinking" or server == "reasoning":
                    server = "sequential-thinking"
                    if tool != "sequentialthinking":
                        tool = "sequentialthinking"

                # SPECIAL TOOL: Handle explicit screenshot requests
                if (server == "macos-use" and tool == "screenshot") or (
                    server == "computer" and tool == "screenshot"
                ):
                    logger.info(
                        "[GRISHA] Internal Decision: Capturing Screenshot for Visual Verification"
                    )
                    try:
                        screenshot_path = await self.take_screenshot()
                        attach_screenshot_next = True
                        verification_history.append(
                            {
                                "tool": f"{server}.{tool}",
                                "args": args,
                                "result": f"Screenshot captured and attached: {screenshot_path}",
                            }
                        )
                        continue
                    except Exception as e:
                        verification_history.append(
                            {
                                "tool": f"{server}.{tool}",
                                "args": args,
                                "result": f"Error taking screenshot: {e}",
                            }
                        )
                        continue

                if "path" in args:
                    original_path = args["path"]
                    resolved_path = shared_context.resolve_path(original_path)
                    if resolved_path != original_path:
                        logger.info(
                            f"[GRISHA] Path auto-corrected: {original_path} -> {resolved_path}"
                        )
                    args["path"] = resolved_path

                logger.info(
                    f"[GRISHA] Internal Verification Tool: {server}.{tool}({args})"
                )

                try:
                    tool_output = await self._call_tool_with_retry(
                        mcp_manager, server, tool, args
                    )
                    if "path" in args and tool_output:
                        shared_context.update_path(args["path"], "verify")
                    logger.info(f"[GRISHA] Tool Output: {str(tool_output)[:500]}...")

                    if server == "computer-use" and tool == "screenshot":
                        try:
                            new_path = None
                            if isinstance(tool_output, dict):
                                new_path = tool_output.get("path")
                            if not new_path and hasattr(tool_output, "content"):
                                for item in getattr(tool_output, "content", []) or []:
                                    txt = getattr(item, "text", "")
                                    if (
                                        isinstance(txt, str)
                                        and "/" in txt
                                        and ".png" in txt
                                    ):
                                        parts = txt.split()
                                        for p in reversed(parts):
                                            if p.startswith("/") and p.endswith(".png"):
                                                new_path = p
                                                break
                                    if new_path:
                                        break

                            if new_path and os.path.exists(new_path):
                                screenshot_path = new_path
                                attach_screenshot_next = True
                            else:
                                # FALLBACK: search notes for verification_artifact notes for this step
                                try:
                                    from ..mcp_manager import mcp_manager
                                    notes_search = await mcp_manager.call_tool(
                                        "notes",
                                        "search_notes",
                                        {"tags": [f"step_{step_id}"], "limit": 5},
                                    )
                                    if (
                                        isinstance(notes_search, dict)
                                        and notes_search.get("result", {}).get("success")
                                    ):
                                        for n in notes_search["result"].get("notes", []):
                                            note_file = n.get("file")
                                            if note_file and os.path.exists(note_file):
                                                try:
                                                    text = open(note_file, "r", encoding="utf-8").read()
                                                    # Find absolute png paths
                                                    import re

                                                    m = re.search(r"(/[^\s']+?\.png)", text)
                                                    if m:
                                                        candidate = m.group(1)
                                                        if os.path.exists(candidate):
                                                            screenshot_path = candidate
                                                            attach_screenshot_next = True
                                                            logger.info(f"[GRISHA] Attached screenshot from note: {candidate}")
                                                            break
                                                except Exception:
                                                    pass
                                except Exception as e:
                                    logger.warning(f"[GRISHA] Could not search notes for artifacts: {e}")
                        except Exception as e:
                            logger.warning(
                                f"[GRISHA] Could not attach refreshed screenshot: {e}"
                            )
                except Exception as e:
                    tool_output = f"Error calling tool: {e}"
                    logger.error(f"[GRISHA] Tool Error: {e}")

                verification_history.append(
                    {
                        "tool": f"{server}.{tool}",
                        "args": args,
                        "result": str(tool_output),
                    }
                )
                continue
            else:
                # OPTIMIZATION: Early exit on high confidence
                confidence = data.get("confidence", 0.5)
                if confidence >= 0.85:
                    logger.info(f"[GRISHA] High confidence ({confidence}), early exit.")

                verification = VerificationResult(
                    step_id=step_id,
                    verified=data.get("verified", False),
                    confidence=confidence,
                    description=data.get("description", ""),
                    issues=data.get("issues", []),
                    voice_message=data.get("voice_message", ""),
                    screenshot_analyzed=screenshot_path is not None,
                )

                self.verifications.append(verification)

                # Save detailed rejection report to memory if verification failed
                if not verification.verified:
                    await self._save_rejection_report(step_id, step, verification)

                return verification

        # SPECIAL CASE: Consent/Approval/Confirmation steps that are organizational, not technical
        # If step action contains keywords like "consent", "approval", "confirm", "agree", "permission"
        # AND we see evidence of Terminal/TextEdit opened (user can respond there), accept it.
        step_action_lower = step.get("action", "").lower()
        consent_keywords = [
            "consent",
            "approval",
            "confirm",
            "agree",
            "permission",
            "згод",
            "підтверд",
            "дозвіл",
        ]
        is_consent_step = any(
            kw in step_action_lower or kw in expected.lower() for kw in consent_keywords
        )

        if is_consent_step and verification_history:
            # Check if Terminal or similar was opened
            opened_app = any(
                "terminal" in str(h).lower() or "textedit" in str(h).lower()
                for h in verification_history
            )
            if opened_app:
                logger.info(
                    "[GRISHA] Detected consent/approval step with Terminal/TextEdit opened. Auto-accepting as user can respond there."
                )
                return VerificationResult(
                    step_id=step_id,
                    verified=True,
                    confidence=0.7,
                    description="Consent step: Terminal/TextEdit opened for user response. Assuming user can provide consent there.",
                    issues=[],
                    voice_message="Консенс крок: термінал відкритий для вводу. Приймаю.",
                )

        logger.warning(f"[GRISHA] Forcing verdict after {max_attempts} attempts")
        success_count = sum(
            1
            for h in verification_history
            if "error" not in str(h.get("result", "")).lower()
        )
        auto_verified = (
            success_count > 0 and success_count >= len(verification_history) // 2
        )

        return VerificationResult(
            step_id=step_id,
            verified=auto_verified,
            confidence=0.3 if auto_verified else 0.0,
            description=f"Auto-verdict after {max_attempts} tool calls. {success_count}/{len(verification_history)} successful. History: {[h.get('tool') for h in verification_history]}",
            issues=["Max attempts reached", "Forced verdict based on tool history"],
            voice_message=(
                f"Автоматична верифікація не пройшла. Успіх: {success_count} з {len(verification_history)}."
                if not auto_verified
                else f"Автоматично підтверджено. {success_count} перевірок успішні."
            ),
        )

    async def _save_rejection_report(
        self, step_id: int, step: Dict[str, Any], verification: VerificationResult
    ) -> None:
        """Save detailed rejection report to memory and notes servers for Atlas and Tetyana to access"""
        from datetime import datetime

        from ..mcp_manager import mcp_manager

        try:
            timestamp = datetime.now().isoformat()

            # Prepare detailed report text
            report_text = f"""GRISHA VERIFICATION REPORT - REJECTED

Step ID: {step_id}
Action: {step.get('action', '')}
Expected: {step.get('expected_result', '')}
Confidence: {verification.confidence}

DESCRIPTION:
{verification.description}

ISSUES FOUND:
{chr(10).join(f'- {issue}' for issue in verification.issues)}

VOICE MESSAGE (Ukrainian):
{verification.voice_message}

Screenshot Analyzed: {verification.screenshot_analyzed}
Timestamp: {timestamp}
"""

            # Save to memory server (for graph/relations)
            try:
                await mcp_manager.call_tool(
                    "memory",
                    "create_entities",
                    {
                        "entities": [
                            {
                                "name": f"grisha_rejection_step_{step_id}",
                                "entityType": "verification_report",
                                "observations": [report_text],
                            }
                        ]
                    },
                )
                logger.info(
                    f"[GRISHA] Rejection report saved to memory for step {step_id}"
                )
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to save to memory: {e}")

            # Save to notes server (for easy text retrieval)
            try:
                await mcp_manager.call_tool(
                    "notes",
                    "create_note",
                    {
                        "title": f"Grisha Rejection - Step {step_id}",
                        "content": report_text,
                        "category": "verification_report",
                        "tags": ["grisha", "rejection", f"step_{step_id}", "failed"],
                    },
                )
                logger.info(
                    f"[GRISHA] Rejection report saved to notes for step {step_id}"
                )
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to save to notes: {e}")

        except Exception as e:
            logger.warning(f"[GRISHA] Failed to save rejection report: {e}")

    async def security_check(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs security check before execution
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        action_str = str(action)
        if self._check_blocklist(action_str):
            return {
                "safe": False,
                "risk_level": "critical",
                "reason": "Command found in blocklist",
                "requires_confirmation": True,
                "voice_message": "УВАГА! Ця команда у чорному списку. Блокую.",
            }

        prompt = AgentPrompts.grisha_security_prompt(str(action))

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        return self._parse_response(response.content)

    async def take_screenshot(self) -> str:
        """
        Takes a screenshot for verification.
        Enhanced for AtlasTrinity:
        - Robust multi-monitor support (Quartz).
        - Active application window focus (AppleScript).
        - Combined context+detail image for GPT-4o Vision.
        """
        import subprocess
        import tempfile
        from datetime import datetime

        from PIL import Image

        from ..config import SCREENSHOTS_DIR

        try:
            Quartz = None
            quartz_available = False
            try:
                import Quartz as _Quartz  # type: ignore

                Quartz = _Quartz
                quartz_available = True
            except Exception as qerr:
                logger.warning(
                    f"[GRISHA] Quartz unavailable for screenshots (will fallback to screencapture): {qerr}"
                )

            desktop_canvas = None
            active_win_img = None

            if quartz_available and Quartz is not None:
                max_displays = 16
                list_result = Quartz.CGGetActiveDisplayList(max_displays, None, None)
                if not list_result or list_result[0] != 0:
                    raise RuntimeError("Quartz display list error")

                active_displays = list_result[1]
                displays_info = []
                for idx, display_id in enumerate(active_displays):
                    bounds = Quartz.CGDisplayBounds(display_id)
                    displays_info.append(
                        {
                            "id": display_id,
                            "sc_index": idx + 1,
                            "x": bounds.origin.x,
                            "y": bounds.origin.y,
                            "width": bounds.size.width,
                            "height": bounds.size.height,
                        }
                    )

                displays_info.sort(key=lambda d: d["x"])
                min_x = min(d["x"] for d in displays_info)
                min_y = min(d["y"] for d in displays_info)
                max_x = max(d["x"] + d["width"] for d in displays_info)
                max_y = max(d["y"] + d["height"] for d in displays_info)

                total_w = int(max_x - min_x)
                total_h = int(max_y - min_y)
                desktop_canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))

                for d in displays_info:
                    fhandle, path = tempfile.mkstemp(suffix=".png")
                    os.close(fhandle)
                    subprocess.run(
                        ["screencapture", "-x", "-D", str(d["sc_index"]), path],
                        capture_output=True,
                    )
                    if os.path.exists(path):
                        try:
                            with Image.open(path) as img:
                                desktop_canvas.paste(
                                    img.copy(),
                                    (int(d["x"] - min_x), int(d["y"] - min_y)),
                                )
                        finally:
                            try:
                                os.unlink(path)
                            except Exception:
                                pass

                logger.info("[GRISHA] Capturing active application window...")
                active_win_path = os.path.join(
                    tempfile.gettempdir(), "grisha_active_win.png"
                )
                try:
                    window_list = Quartz.CGWindowListCopyWindowInfo(
                        Quartz.kCGWindowListOptionOnScreenOnly
                        | Quartz.kCGWindowListExcludeDesktopElements,
                        Quartz.kCGNullWindowID,
                    )
                    front_win_id = None
                    for window in window_list:
                        if window.get("kCGWindowLayer") == 0:
                            front_win_id = window.get("kCGWindowNumber")
                            break

                    if front_win_id:
                        subprocess.run(
                            [
                                "screencapture",
                                "-l",
                                str(front_win_id),
                                "-x",
                                active_win_path,
                            ],
                            capture_output=True,
                        )
                except Exception as win_err:
                    logger.warning(f"Failed to detect active window ID: {win_err}")

                if os.path.exists(active_win_path):
                    try:
                        with Image.open(active_win_path) as img:
                            active_win_img = img.copy()
                    except Exception:
                        pass
                    finally:
                        try:
                            os.unlink(active_win_path)
                        except Exception:
                            pass
            else:
                display_imgs = []
                consecutive_failures = 0
                for di in range(1, 17):
                    fhandle, path = tempfile.mkstemp(suffix=".png")
                    os.close(fhandle)
                    try:
                        res = subprocess.run(
                            ["screencapture", "-x", "-D", str(di), path],
                            capture_output=True,
                        )
                        if res.returncode == 0 and os.path.exists(path):
                            with Image.open(path) as img:
                                display_imgs.append(img.copy())
                            consecutive_failures = 0
                        else:
                            consecutive_failures += 1
                    finally:
                        try:
                            if os.path.exists(path):
                                os.unlink(path)
                        except Exception:
                            pass

                    if display_imgs and consecutive_failures >= 2:
                        break

                if not display_imgs:
                    tmp_full = os.path.join(
                        tempfile.gettempdir(),
                        f"grisha_full_{datetime.now().strftime('%H%M%S')}.png",
                    )
                    subprocess.run(
                        ["screencapture", "-x", tmp_full], capture_output=True
                    )
                    if os.path.exists(tmp_full):
                        try:
                            with Image.open(tmp_full) as img:
                                desktop_canvas = img.copy()
                        finally:
                            try:
                                os.unlink(tmp_full)
                            except Exception:
                                pass
                else:
                    total_w = sum(img.width for img in display_imgs)
                    max_h = max(img.height for img in display_imgs)
                    desktop_canvas = Image.new("RGB", (total_w, max_h), (0, 0, 0))
                    x_off = 0
                    for img in display_imgs:
                        desktop_canvas.paste(img, (x_off, 0))
                        x_off += img.width

            if desktop_canvas is None:
                raise RuntimeError("Failed to capture desktop canvas")

            target_w = 2048
            scale = target_w / max(1, desktop_canvas.width)
            dt_h = int(desktop_canvas.height * scale)
            desktop_small = desktop_canvas.resize(
                (target_w, max(1, dt_h)), Image.LANCZOS
            )

            final_h = desktop_small.height
            if active_win_img:
                win_scale = target_w / max(1, active_win_img.width)
                win_h = int(active_win_img.height * win_scale)
                final_h += win_h + 20
                final_canvas = Image.new("RGB", (target_w, final_h), (30, 30, 30))
                final_canvas.paste(desktop_small, (0, 0))
                final_canvas.paste(
                    active_win_img.resize((target_w, max(1, win_h)), Image.LANCZOS),
                    (0, desktop_small.height + 20),
                )
            else:
                final_canvas = desktop_small

            final_path = os.path.join(
                str(SCREENSHOTS_DIR),
                f"grisha_vision_{datetime.now().strftime('%H%M%S')}.jpg",
            )
            final_canvas.save(final_path, "JPEG", quality=85)
            logger.info(f"[GRISHA] Vision composite saved: {final_path}")
            return final_path

        except Exception as e:
            logger.warning(
                f"Combined screenshot failed: {e}. Falling back to simple grab."
            )
            try:
                from PIL import ImageGrab

                screenshot = ImageGrab.grab(all_screens=True)
                temp_path = os.path.join(
                    str(SCREENSHOTS_DIR),
                    f"grisha_verify_fallback_{datetime.now().strftime('%H%M%S')}.jpg",
                )
                screenshot.save(temp_path, "JPEG", quality=80)
                return temp_path
            except Exception:
                return ""

    def get_voice_message(self, action: str, **kwargs) -> str:
        """Generates short message for TTS"""
        messages = {
            "verified": "Тетяно, я бачу що завдання виконано. Можеш продовжувати.",
            "failed": "Тетяно, результат не відповідає очікуванню.",
            "blocked": "УВАГА! Ця дія небезпечна. Блокую виконання.",
            "checking": "Перевіряю результат...",
            "approved": "Підтверджую. Можна продовжувати.",
        }
        return messages.get(action, "")

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        import json

        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        return {"raw": content}
