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

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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
from .base_agent import BaseAgent  # noqa: E402


@dataclass
class VerificationResult:
    """Verification result"""

    step_id: str
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


class Grisha(BaseAgent):
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
            final_model = agent_config.get("vision_model") or os.getenv("VISION_MODEL", "gpt-4o")

        self.llm = CopilotLLM(model_name=final_model, vision_model_name=final_model)
        self.temperature = agent_config.get("temperature", 0.3)

        # Load dangerous commands from config with fallback to hardcoded BLOCKLIST
        self.dangerous_commands = security_config.get("dangerous_commands", self.BLOCKLIST)
        self.verifications: list = []

        # OPTIMIZATION: Strategy cache to avoid redundant LLM calls
        self._strategy_cache = {}

        # Reasoner Model (Raptor-Mini) for Strategy Planning
        # Default to raptor-mini, or use from config/env
        strategy_model = agent_config.get("strategy_model") or os.getenv(
            "STRATEGY_MODEL", "raptor-mini"
        )
        self.strategist = CopilotLLM(model_name=strategy_model)
        logger.info(f"[GRISHA] Initialized with Vision={final_model}, Strategy={strategy_model}")

    async def _plan_verification_strategy(
        self,
        step_action: str,
        expected_result: str,
        context: dict,
        goal_context: str = "",
    ) -> str:
        """
        Uses Raptor-Mini (MSP Reasoning) to create a robust verification strategy.
        OPTIMIZATION: Caches strategies by step type to avoid redundant LLM calls.
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

        # OPTIMIZATION: Check cache first
        cache_key = f"{step_action[:50]}:{expected_result[:50]}"
        if cache_key in self._strategy_cache:
            logger.info(f"[GRISHA] Using cached strategy for: {cache_key[:30]}...")
            return self._strategy_cache[cache_key]

        prompt = AgentPrompts.grisha_strategy_prompt(
            step_action, expected_result, context, goal_context=goal_context
        )

        # Get available capabilities to inform the strategist
        capabilities = self._get_environment_capabilities()
        system_msg = AgentPrompts.grisha_strategist_system_prompt(capabilities)
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

    def _get_environment_capabilities(self) -> str:
        """
        Collects raw facts about the environment to inform the strategist.
        No heuristics here—just data for the LLM to reason about.
        """
        try:
            from ..mcp_manager import mcp_manager
            servers_cfg = getattr(mcp_manager, "config", {}).get("mcpServers", {})
            active_servers = [s for s, cfg in servers_cfg.items() if not (cfg or {}).get("disabled")]
            
            swift_servers = []
            for s in active_servers:
                cfg = servers_cfg.get(s, {})
                cmd = (cfg or {}).get("command", "") or ""
                if "swift" in s.lower() or "macos" in s.lower() or (isinstance(cmd, str) and "swift" in cmd.lower()):
                    swift_servers.append(s)
        except Exception:
            active_servers = []
            swift_servers = []

        vision_model = (getattr(self.llm, "model_name", "") or "unknown").lower()
        is_powerful = any(x in vision_model for x in ("gpt-4o", "vision", "claude-3-5"))

        info = [
            f"Active MCP Realms: {', '.join(active_servers)}",
            f"Native Swift Servers: {', '.join(swift_servers)} (Preferred for OS control)",
            f"Vision Model: {vision_model} ({'High-Performance' if is_powerful else 'Standard'})",
            f"Timezone: {datetime.now().astimezone().tzname()}",
            "Capabilities: Full UI Traversal, OCR, Terminal, Filesystem, Apple Productivity Apps integration."
        ]
        return "\n".join(info)


    def _summarize_ui_data(self, raw_data: str) -> str:
        """
        Intelligently extracts the 'essence' of UI traversal data locally.
        Reduces thousands of lines of JSON to a concise list of key interactive elements.
        """
        import json
        if not raw_data or not isinstance(raw_data, str) or not (raw_data.strip().startswith('{') or raw_data.strip().startswith('[')):
            return raw_data

        try:
            data = json.loads(raw_data)
            # Find the list of elements (robust to various nesting levels)
            elements = []
            if isinstance(data, list):
                elements = data
            elif isinstance(data, dict):
                # Search common keys: 'elements', 'result', etc.
                if "elements" in data: elements = data["elements"]
                elif "result" in data and isinstance(data["result"], dict):
                    elements = data["result"].get("elements", [])
                elif "result" in data and isinstance(data["result"], list):
                    elements = data["result"]

            if not elements or not isinstance(elements, list):
                return raw_data[:2000] # Fallback to truncation

            summary_items = []
            for el in elements:
                if not isinstance(el, dict): continue
                
                # Filter: Only care about visible or important elements to save tokens
                if el.get("isVisible") is False and not el.get("label") and not el.get("title"):
                    continue
                
                role = el.get("role", "element")
                label = el.get("label") or el.get("title") or el.get("description") or el.get("help")
                value = el.get("value") or el.get("stringValue")
                
                # Only include if it has informative content
                if label or value or role in ["AXButton", "AXTextField", "AXTextArea", "AXCheckBox"]:
                    item = f"[{role}"
                    if label: item += f": '{label}'"
                    if value: item += f", value: '{value}'"
                    item += "]"
                    summary_items.append(item)
            
            summary = " | ".join(summary_items)
            
            # Final check: if summary is still somehow empty but we had elements, 
            # maybe we were too aggressive. Provide a tiny slice of raw.
            if not summary and elements:
                return f"UI Tree Summary: {len(elements)} elements found. Samples: {str(elements[:2])}"
                
            return f"UI Summary ({len(elements)} elements): " + summary
            
        except Exception as e:
            logger.debug(f"[GRISHA] UI summarization failed (falling back to truncation): {e}")
            return raw_data[:3000]

    async def _fetch_execution_trace(self, step_id: str, task_id: Optional[str] = None) -> str:
        """
        Fetches the raw tool execution logs from the database for a given step.
        This serves as the 'single source of truth' for verification.
        """
        try:
            from ..mcp_manager import mcp_manager
            
            # Query db for tool executions related to this step, including the status from task_steps
            if task_id:
                sql = """
                    SELECT te.tool_name, te.arguments, te.result, ts.status as step_status, te.created_at 
                    FROM tool_executions te
                    JOIN task_steps ts ON te.step_id = ts.id
                    WHERE ts.sequence_number = :seq AND ts.task_id = :task_id
                    ORDER BY te.created_at DESC 
                    LIMIT 5;
                """
                params = {"seq": str(step_id), "task_id": task_id}
            else:
                sql = """
                    SELECT te.tool_name, te.arguments, te.result, ts.status as step_status, te.created_at 
                    FROM tool_executions te
                    JOIN task_steps ts ON te.step_id = ts.id
                    WHERE ts.sequence_number = :seq 
                    ORDER BY te.created_at DESC 
                    LIMIT 5;
                """
                params = {"seq": str(step_id)}
            
            rows = await mcp_manager.query_db(sql, params)
            
            if not rows:
                return "No DB records found for this step. (Command might not have been logged yet or step ID mismatch)."

            trace = "\n--- TECHNICAL EXECUTION TRACE (FROM DB) ---\n"
            for row in rows:
                tool = row.get("tool_name", "unknown")
                args = row.get("arguments", {})
                res = str(row.get("result", ""))
                status = row.get("step_status", "unknown")
                
                # Truncate result for token saving
                if len(res) > 2000:
                    res = res[:2000] + "...(truncated)"
                
                trace += f"Tool: {tool}\nArgs: {args}\nStep Status (from Tetyana): {status}\nResult: {res or '(No output - Silent Success)'}\n-----------------------------------\n"
            
            return trace

        except Exception as e:
            logger.warning(f"[GRISHA] Failed to fetch execution trace: {e}")
            return f"Error fetching trace: {e}"

    async def verify_step(
        self,
        step: Dict[str, Any],
        result: Any,
        screenshot_path: Optional[str] = None,
        goal_context: str = "",
        task_id: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verifies the result of step execution using Vision and MCP Tools
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

        from ..mcp_manager import mcp_manager  # noqa: E402

        step_id = step.get("id", 0)
        expected = step.get("expected_result", "")

        # PRIORITY: Use MCP tools first, screenshots only when explicitly needed
        # Only take screenshot if explicitly requested or if visual verification is clearly needed
        visual_verification_needed = (
            "visual" in expected.lower()
            or "screenshot" in expected.lower()
            or "ui" in expected.lower()
            or "interface" in expected.lower()
            or "window" in expected.lower()
        )
        
        # RELAXATION: Don't demand legal/intent verification for technical tasks
        # unless keywords are present
        requires_legal_check = (
             "legal" in expected.lower()
             or "intent" in expected.lower()
             or "compliance" in expected.lower()
             or "policy" in expected.lower()
        )

        if (
            not visual_verification_needed
            or not screenshot_path
            or not isinstance(screenshot_path, str)
            or not os.path.exists(screenshot_path)
        ):
            screenshot_path = None

        # If we don't already have a screenshot path, try to find artifacts saved by Tetyana
        # (Simplified: Reliance on shared_context or direct params mostly)
        if not screenshot_path:
             pass

        context_info = shared_context.to_dict()

        if hasattr(result, "result") and not isinstance(result, dict):
            actual_raw = str(result.result)
        elif isinstance(result, dict):
            actual_raw = str(result.get("result", result.get("output", "")))
        else:
            actual_raw = str(result)

        # NEW: Intelligent local summarization instead of simple truncation
        actual = self._summarize_ui_data(actual_raw)
        
        # Inject tool execution details to prove execution to the LLM
        tool_proof = ""
        if hasattr(result, "tool_call") and result.tool_call:
             tool_proof = f"\n\n[PROOF OF EXECUTION]\nTool: {result.tool_call.get('name')}\nArgs: {result.tool_call.get('args')}\n"
        elif isinstance(result, dict) and result.get("tool_call"):
             tc = result["tool_call"]
             tool_proof = f"\n\n[PROOF OF EXECUTION]\nTool: {tc.get('name')}\nArgs: {tc.get('args')}\n"
        
        actual += tool_proof

        # Double safety truncation for the final string sent to LLM
        if len(actual) > 8000:
            actual = actual[:8000] + "...(truncated for brevity)"

        # 1. PLAN STRATEGY with Raptor-Mini
        strategy_context = await self._plan_verification_strategy(
            step.get("action", ""),
            expected,
            context_info,
            goal_context=goal_context or shared_context.get_goal_context(),
        )

        # 2. FETCH TECHNICAL TRACE FROM DB (The "Truth")
        technical_trace = await self._fetch_execution_trace(str(step_id), task_id=task_id)
        
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
                technical_trace=technical_trace,  # Pass the trace
                goal_context=goal_context or shared_context.get_goal_context(),
                tetyana_thought=getattr(result, "thought", "")
                if not isinstance(result, dict)
                else result.get("thought", ""),
            )
            content.append({"type": "text", "text": prompt_text})

            if (
                screenshot_path
                and os.path.exists(screenshot_path)
                and (attempt == 0 or attach_screenshot_next)
            ):
                with open(screenshot_path, "rb") as f:
                    img_bytes = f.read()
                    
                # OPTIMIZATION: If image is too large (> 500KB), compress it for the prompt
                if len(img_bytes) > 500 * 1024:
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(img_bytes))
                        # Convert to RGB if necessary
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # Limit max dimensions to 1024px for faster/cheaper vision
                        img.thumbnail((1024, 1024), Image.LANCZOS)
                        
                        output = io.BytesIO()
                        img.save(output, format="JPEG", quality=70, optimize=True)
                        img_bytes = output.getvalue()
                        logger.info(f"[GRISHA] Compressed screenshot for prompt: {len(img_bytes)} bytes")
                    except Exception as e:
                        logger.warning(f"[GRISHA] Failed to compress screenshot: {e}")

                image_data = base64.b64encode(img_bytes).decode("utf-8")

                mime = "image/jpeg" # We force JPEG after compression
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
            logger.info(f"[GRISHA] Raw LLM Response: {response.content}")
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

                if server in ["terminal", "macos-use", "computer", "system", "local", "sh", "bash"]:
                    if any(t in tool.lower() for t in ["terminal", "run", "execute", "shell", "exec", "command"]):
                        server = "macos-use"
                        tool = "execute_command"
                    elif any(t in tool.lower() for t in ["screenshot", "take_screenshot", "capture"]):
                        server = "macos-use"
                        tool = "macos-use_take_screenshot"
                    elif any(t in tool.lower() for t in ["vision", "analyze", "ocr", "scan"]):
                        server = "macos-use"
                        tool = "macos-use_analyze_screen"
                    elif any(t in tool.lower() for t in ["time", "clock", "date"]):
                        server = "macos-use"
                        tool = "macos-use_get_time"
                    elif any(t in tool.lower() for t in ["fetch", "url", "scrape", "wget", "curl"]):
                        server = "macos-use"
                        tool = "macos-use_fetch_url"
                    elif any(t in tool.lower() for t in ["calendar", "event"]):
                        server = "macos-use"
                        tool = "macos-use_calendar_events"
                    elif any(t in tool.lower() for t in ["reminder"]):
                        server = "macos-use"
                        tool = "macos-use_reminders"
                    elif any(t in tool.lower() for t in ["note"]):
                        server = "macos-use"
                        tool = "macos-use_notes_list_folders"
                    elif any(t in tool.lower() for t in ["mail", "email"]):
                        server = "macos-use"
                        tool = "macos-use_mail_read_inbox"
                    elif any(t in tool.lower() for t in ["applescript", "script"]):
                        server = "macos-use"
                        tool = "macos-use_run_applescript"
                    elif any(t in tool.lower() for t in ["finder", "file_list", "ls", "dir", "path"]):
                        server = "macos-use"
                        tool = "macos-use_finder_list_files"
                    elif any(t in tool.lower() for t in ["spotlight", "mdfind", "search"]):
                        server = "macos-use"
                        tool = "macos-use_spotlight_search"
                    elif any(t in tool.lower() for t in ["notify", "notification", "alert"]):
                        server = "macos-use"
                        tool = "macos-use_send_notification"
                    elif tool.startswith("macos-use_"):
                        server = "macos-use"
                    
                    if tool in ["ls", "list", "dir"] and server == "macos-use" and "path" not in args:
                        tool = "execute_command"
                        args = {"command": f"ls -la {args.get('path', '.')}"}
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

                logger.info(f"[GRISHA] Internal Verification Tool: {server}.{tool}({args})")

                try:
                    tool_output = await mcp_manager.dispatch_tool(f"{server}.{tool}", args)
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
                                    if isinstance(txt, str) and "/" in txt and ".png" in txt:
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
                                pass # No fallback notes search 
                        except Exception as e:
                            logger.warning(f"[GRISHA] Could not attach refreshed screenshot: {e}")
                except Exception as e:
                    tool_output = f"Error calling tool: {e}"
                    logger.error(f"[GRISHA] Tool Error: {e}")

                # Truncate large tool outputs to prevent context window overflow
                result_str = str(tool_output)
                if len(result_str) > 1000:
                    result_str = result_str[:1000] + "...(truncated)"

                verification_history.append(
                    {
                        "tool": f"{server}.{tool}",
                        "args": args,
                        "result": result_str,
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
                    description=data.get("description") or f"No description provided. Raw data: {data}",
                    issues=data.get("issues", []),
                    voice_message=data.get("voice_message", ""),
                    screenshot_analyzed=screenshot_path is not None,
                )

                self.verifications.append(verification)

                # Save detailed rejection report to memory if verification failed
                if not verification.verified:
                    await self._save_rejection_report(step_id, step, verification, task_id=task_id)

                return verification

        # SPECIAL CASE: Consent/Approval/Confirmation steps that are organizational, not technical
        # If step action contains keywords like "consent", "approval", "confirm", "agree", "permission"
        # AND we see evidence of Terminal/TextEdit opened (user can respond there), accept it.
        step_action_lower = step.get("action", "").lower()
        consent_keywords = [
            "ask user",
            "request user consent",
            "await user approval",
            "get user confirmation",
            "confirm with user",
            "згод",
            "підтверд",
            "дозвіл",
        ]
        is_manual_consent = any(
            kw in step_action_lower for kw in consent_keywords
        ) or step.get("requires_consent", False) is True

        if is_manual_consent and verification_history:
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
            1 for h in verification_history if "error" not in str(h.get("result", "")).lower()
        )
        auto_verified = success_count > 0 and success_count >= len(verification_history) // 2

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

    async def analyze_failure(self, step: Dict[str, Any], error: str, context: dict = None) -> Dict[str, Any]:
        """
        Analyzes a failure reported by Tetyana or Orchestrator.
        Returns constructive feedback for a retry.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        step_id = step.get("id", "unknown")
        context_data = context or shared_context.to_dict()
        
        prompt = AgentPrompts.grisha_failure_analysis_prompt(
            str(step),
            error,
            context_data,
            plan_context=step.get("full_plan", "")
        )
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            analysis = self._parse_response(response.content)
            
            logger.info(f"[GRISHA] Failure Analysis for step {step_id}: {analysis.get('root_cause')}")
            
            return {
                "step_id": step_id,
                "analysis": analysis,
                "feedback_text": f"GRISHA FEEDBACK: {analysis.get('root_cause')}\nADVICE: {analysis.get('technical_advice')}",
                "voice_message": analysis.get("voice_message")
            }
        except Exception as e:
            logger.error(f"[GRISHA] Failure analysis failed: {e}")
            return {
                "step_id": step_id,
                "analysis": {},
                "feedback_text": f"GRISHA FEEDBACK: Unknown error analysis. Original error: {error}",
                "voice_message": "Я не зміг проаналізувати помилку, але раджу спробувати ще раз."
            }

    async def _save_rejection_report(
        self,
        step_id: str,
        step: Dict[str, Any],
        verification: VerificationResult,
        task_id: Optional[str] = None,
    ) -> None:
        """Save detailed rejection report to memory and notes servers for Atlas and Tetyana to access"""
        from datetime import datetime  # noqa: E402

        from ..knowledge_graph import knowledge_graph  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402
        from ..message_bus import AgentMsg, MessageType, message_bus  # noqa: E402

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
                await mcp_manager.dispatch_tool(
                    "memory.create_entities",
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
                logger.info(f"[GRISHA] Rejection report saved to memory for step {step_id}")
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to save to memory: {e}")
 
            # Save to filesystem (for easy text retrieval)
            try:
                reports_dir = os.path.expanduser("~/.config/atlastrinity/reports")
                os.makedirs(reports_dir, exist_ok=True)
                
                filename = f"rejection_step_{step_id}_{int(datetime.now().timestamp())}.md"
                file_path = os.path.join(reports_dir, filename)
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                    
                logger.info(f"[GRISHA] Rejection report saved to filesystem: {file_path}")
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to save report to filesystem: {e}")

            # Save to knowledge graph (Structured Semantic Memory)
            try:
                node_id = f"rejection:step_{step_id}_{int(datetime.now().timestamp())}"
                await knowledge_graph.add_node(
                    node_type="CONCEPT",
                    node_id=node_id,
                    attributes={
                        "type": "verification_rejection",
                        "step_id": str(step_id),
                        "issues": "; ".join(verification.issues) if isinstance(verification.issues, list) else str(verification.issues),
                        "description": str(verification.description),
                        "timestamp": timestamp
                    }
                )
                # Link to the task (use task_id if provided)
                source_id = f"task:{task_id}" if task_id else f"task:rejection_{step_id}"
                await knowledge_graph.add_edge(
                    source_id=source_id,
                    target_id=node_id,
                    relation="REJECTED"
                )
                logger.info(f"[GRISHA] Rejection node added to Knowledge Graph for step {step_id}")
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to update Knowledge Graph: {e}")

            # Send to Message Bus (Real-time typed communication)
            try:
                msg = AgentMsg(
                    from_agent="grisha",
                    to_agent="tetyana",
                    message_type=MessageType.REJECTION,
                    payload={
                        "step_id": str(step_id),
                        "issues": verification.issues,
                        "description": verification.description,
                        "remediation": getattr(verification, "remediation_suggestions", [])
                    },
                    step_id=str(step_id)
                )
                await message_bus.send(msg)
                logger.info(f"[GRISHA] Rejection message sent to Tetyana via Message Bus")
            except Exception as e:
                logger.warning(f"[GRISHA] Failed to send message to bus: {e}")

        except Exception as e:
            logger.warning(f"[GRISHA] Failed to save rejection report: {e}")

    async def security_check(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs security check before execution
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

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
        import subprocess  # noqa: E402
        import tempfile  # noqa: E402
        from datetime import datetime  # noqa: E402

        from PIL import Image  # noqa: E402

        from ..config import SCREENSHOTS_DIR  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        # 1. Try Native Swift MCP first (fastest, most reliable)
        try:
             # Check if macos-use is active
             if "macos-use" in mcp_manager.config.get("mcpServers", {}):
                 result = await mcp_manager.call_tool("macos-use", "macos-use_take_screenshot", {})
                 
                 # Result might be a dict with content->text (base64) OR direct base64 string depending on how call_tool processes it
                 base64_img = None
                 if isinstance(result, dict) and "content" in result:
                     for item in result["content"]:
                         if item.get("type") == "text":
                             base64_img = item.get("text")
                             break
                 elif hasattr(result, "content"): # prompt object
                      if len(result.content) > 0 and hasattr(result.content[0], "text"):
                           base64_img = result.content[0].text
                           
                 if base64_img:
                      # Save to file for consistency with rest of pipeline
                      os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                      path = os.path.join(SCREENSHOTS_DIR, f"vision_mcp_{timestamp}.jpg")
                      
                      with open(path, "wb") as f:
                          f.write(base64.b64decode(base64_img))
                          
                      logger.info(f"[GRISHA] Screenshot taken via MCP macos-use: {path}")
                      return path
        except Exception as e:
            logger.warning(f"[GRISHA] MCP screenshot failed, falling back to local Quartz: {e}")

        # 2. Local Fallback (Quartz/Screencapture)
        try:
            Quartz = None
            quartz_available = False
            try:
                import Quartz as _Quartz  # type: ignore  # noqa: E402

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
                active_win_path = os.path.join(tempfile.gettempdir(), "grisha_active_win.png")
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
                    subprocess.run(["screencapture", "-x", tmp_full], capture_output=True)
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
            desktop_small = desktop_canvas.resize((target_w, max(1, dt_h)), Image.LANCZOS)

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
            logger.warning(f"Combined screenshot failed: {e}. Falling back to simple grab.")
            try:
                from PIL import ImageGrab  # noqa: E402

                screenshot = ImageGrab.grab(all_screens=True)
                temp_path = os.path.join(
                    str(SCREENSHOTS_DIR),
                    f"grisha_verify_fallback_{datetime.now().strftime('%H%M%S')}.jpg",
                )
                screenshot.save(temp_path, "JPEG", quality=80)
                return temp_path
            except Exception:
                return ""

    async def audit_vibe_fix(
        self,
        error: str,
        vibe_report: str,
        context: dict = None,
        task_id: str = None
    ) -> Dict[str, Any]:
        """
        Audits a proposed fix from Vibe AI before execution.
        Uses advanced reasoning to ensure safety and correctness.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        context_data = context or shared_context.to_dict()
        
        # Fetch technical trace for grounding
        technical_trace = ""
        try:
            # We use the current step if available in context, or try to infer
            step_id = context_data.get("current_step_id", "unknown")
            technical_trace = await self._fetch_execution_trace(str(step_id), task_id=task_id)
        except Exception as e:
            logger.warning(f"[GRISHA] Could not fetch trace for audit: {e}")

        prompt = AgentPrompts.grisha_vibe_audit_prompt(
            error,
            vibe_report,
            context_data,
            technical_trace=technical_trace
        )
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        try:
            logger.info(f"[GRISHA] Auditing Vibe's proposed fix...")
            response = await self.llm.ainvoke(messages)
            audit_result = self._parse_response(response.content)
            
            logger.info(f"[GRISHA] Audit Verdict: {audit_result.get('audit_verdict', 'REJECT')}")
            return audit_result
        except Exception as e:
            logger.error(f"[GRISHA] Vibe audit failed: {e}")
            return {
                "audit_verdict": "REJECT",
                "reasoning": f"Audit failed due to technical error: {str(e)}",
                "voice_message": "Я не зміг перевірити запропоноване виправлення через технічну помилку."
            }

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

