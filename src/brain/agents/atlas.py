"""
Atlas - The Strategist

Role: Strategic analysis, plan formulation, task delegation
Voice: Dmytro (male)
Model: GPT-4.1 / GPT-5 mini
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

current_dir = os.path.dirname(os.path.abspath(__file__))
# Check root (Dev: src/brain/agents -> root)
root_dev = os.path.join(current_dir, "..", "..", "..")
# Check resources (Prod: brain/agents -> Resources)
root_prod = os.path.join(current_dir, "..", "..")

for r in [root_dev, root_prod]:
    abs_r = os.path.abspath(r)
    if abs_r not in sys.path:
        sys.path.insert(0, abs_r)

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from providers.copilot import CopilotLLM  # noqa: E402
from src.brain.agents.base_agent import BaseAgent  # noqa: E402
from src.brain.config_loader import config  # noqa: E402
from src.brain.context import shared_context  # noqa: E402
from src.brain.logger import logger  # noqa: E402
from src.brain.memory import long_term_memory  # noqa: E402
from src.brain.prompts import AgentPrompts  # noqa: E402
from src.brain.prompts.atlas_chat import generate_atlas_chat_prompt, generate_atlas_solo_task_prompt  # noqa: E402


@dataclass
class TaskPlan:
    """Execution plan structure"""

    id: str
    goal: str
    steps: list[dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, active, completed, failed
    context: dict[str, Any] = field(default_factory=dict)


class Atlas(BaseAgent):
    """
    Atlas - The Strategist

    Functions:
    - User context analysis
    - ChromaDB search (historical experience)
    - Global strategy formulation
    - Execution plan creation
    - Task delegation to Tetyana
    """

    NAME = AgentPrompts.ATLAS["NAME"]
    DISPLAY_NAME = AgentPrompts.ATLAS["DISPLAY_NAME"]
    VOICE = AgentPrompts.ATLAS["VOICE"]
    COLOR = AgentPrompts.ATLAS["COLOR"]
    SYSTEM_PROMPT = AgentPrompts.ATLAS["SYSTEM_PROMPT"]

    def __init__(self, model_name: str = "gpt-4o"):
        # Get model config (config.yaml > parameter > env variables)
        agent_config = config.get_agent_config("atlas")
        final_model = model_name
        
        # If default is passed but config has something else, prefer config
        config_model = agent_config.get("model")
        if config_model:
            final_model = config_model
        elif model_name == "gpt-4o": # matching default arg
             # Try env
             final_model = os.getenv("COPILOT_MODEL", "gpt-4o")

        self.llm = CopilotLLM(model_name=final_model)

        # Optimization: Tool Cache
        self._cached_info_tools = []
        self._last_tool_refresh = 0
        self._refresh_interval = 1800  # 30 minutes
        self.temperature = agent_config.get("temperature", 0.7)
        self.current_plan: TaskPlan | None = None
        self.history: list[dict[str, Any]] = []

    async def _get_mcp_capabilities_context(self) -> dict[str, Any]:
        """
        Analyzes available MCP servers and their capabilities.
        Returns structured data for intelligent step planning.
        """
        from ..mcp_manager import mcp_manager
        from ..mcp_registry import SERVER_CATALOG, get_tool_names_for_server

        mcp_config = mcp_manager.config.get("mcpServers", {})
        status = mcp_manager.get_status()
        connected = set(status.get("connected_servers", []))

        capabilities: dict[str, Any] = {
            "active_servers": [],
            "server_capabilities": {},
            "tool_availability": {},
            "recommendations": [],
        }

        for server_name, cfg in mcp_config.items():
            if cfg and cfg.get("disabled"):
                continue

            server_info = SERVER_CATALOG.get(server_name, {})
            is_connected = server_name in connected

            capabilities["active_servers"].append(
                {
                    "name": server_name,
                    "tier": server_info.get("tier", 4),
                    "category": server_info.get("category", "unknown"),
                    "description": server_info.get("description", ""),
                    "connected": is_connected,
                    "key_tools": server_info.get("key_tools", [])[:5],
                    "when_to_use": server_info.get("when_to_use", ""),
                }
            )

            if is_connected:
                capabilities["tool_availability"][server_name] = get_tool_names_for_server(
                    server_name
                )[:10]

        capabilities["active_servers"].sort(key=lambda x: x["tier"])
        logger.info(
            f"[ATLAS] MCP Infrastructure: {len(capabilities['active_servers'])} servers available"
        )
        return capabilities

    async def analyze_request(
        self,
        user_request: str,
        context: dict[str, Any] | None = None,
        history: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Analyzes user request: determines intent (chat vs task)"""

        user_request.lower().strip()

        # No more hardcoded heuristics. The system relies on its 'brain' (LLM) to classify intent.
        # This prevents robotic, predictable responses to keywords like 'привіт'.

        prompt = AgentPrompts.atlas_intent_classification_prompt(
            user_request, str(context or "None"), str(history or "None")
        )
        system_prompt = self.SYSTEM_PROMPT.replace("{{CONTEXT_SPECIFIC_DOCTRINE}}", "")
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            analysis = self._parse_response(cast(str, response.content))

            # Ensure we have a valid intent even if the AI is vague
            if not analysis.get("intent"):
                analysis["intent"] = "chat"
            if not analysis.get("enriched_request"):
                analysis["enriched_request"] = user_request

            return analysis
        except Exception as e:
            logger.error(f"Intent detection LLM failed: {e}")
            return {
                "intent": "chat",
                "reason": f"System fallback due to technical issue: {e}",
                "enriched_request": user_request,
                "complexity": "low",
                "use_deep_persona": False,
                "initial_response": None,  # Force falling back to atlas.chat() for dynamic response
            }

    async def evaluate_deviation(
        self, current_step: dict, proposed_deviation: str, full_plan: list
    ) -> dict:
        """
        Evaluates a strategic deviation proposed by Tetyana.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = AgentPrompts.atlas_deviation_evaluation_prompt(
            str(current_step),
            proposed_deviation,
            context=json.dumps(shared_context.to_dict()),
            full_plan=str(full_plan),
        )

        # Strip system prompt placeholder
        system_prompt = self.SYSTEM_PROMPT.replace("{{CONTEXT_SPECIFIC_DOCTRINE}}", "")

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        try:
            response = await self.llm.ainvoke(messages)
            evaluation = self._parse_response(cast(str, response.content))
            logger.info(f"[ATLAS] Deviation Evaluation: {evaluation.get('approved')}")
            return evaluation
        except Exception as e:
            logger.error(f"[ATLAS] Evaluation failed: {e}")
            return {
                "approved": False,
                "reason": "Evaluation failed",
                "voice_message": "Помилка оцінки.",
            }

    async def chat(
        self,
        user_request: str,
        history: list[Any] | None = None,
        use_deep_persona: bool = False,
        intent: str = "chat",
    ) -> str:
        """
        Omni-Knowledge Chat Mode.
        Integrates Graph Memory, Vector Memory, and System Context for deep awareness.
        """
        import time

        from langchain_core.messages import HumanMessage, SystemMessage

        from ..mcp_manager import mcp_manager

        # 1. Fast-Path: Simple Greeting Detection
        # If it's a short greeting, skip expensive lookups and tool discovery
        request_lower = user_request.lower().strip().rstrip(".?!")
        is_info_query = any(
            kw in request_lower
            for kw in [
                "погода", "weather", "прогноз", "температура",
                "новини", "news", "ціна", "price", "курс",
                "який час", "what time", "скільки", "системн",
                "версія", "version", "файл", "file", "знайди", "find",
                "пошук", "search", "покажи", "шукай", "розкажи про", "прочитай",
            ]
        )
        is_simple_chat = (
            len(user_request.split()) < 5
            and any(
                g in request_lower
                for g in ["привіт", "хай", "hello", "hi", "атлас", "atlas", "як справи", "що ти", "дякую", "окей", "ок"]
            )
            and not is_info_query
        )

        graph_context = ""
        vector_context = ""
        system_status = ""
        available_tools_info = []

        # 2. Parallel Data Fetching: Graph, Vector, and Tools
        if not is_simple_chat or intent == "solo_task":
            logger.info(f"[ATLAS CHAT] Fetching context in parallel for ({intent}): {user_request[:30]}...")
            
            async def get_graph():
                try:
                    res = await mcp_manager.call_tool("memory", "search_nodes", {"query": user_request})
                    if isinstance(res, dict) and "results" in res:
                        return "\n".join([f"Entity: {e.get('name')} | Info: {'; '.join(e.get('observations', [])[:2])}" for e in res.get("results", [])[:2]])
                except Exception: return ""
                return ""

            async def get_vector():
                v_ctx = ""
                try:
                    if long_term_memory.available:
                        # Vector recall in thread to avoid blocking event loop
                        tasks_res = await asyncio.to_thread(long_term_memory.recall_similar_tasks, user_request, n_results=1)
                        if tasks_res: v_ctx += "\nPast Strategy: " + tasks_res[0]["document"][:200]
                        conv_res = await asyncio.to_thread(long_term_memory.recall_similar_conversations, user_request, n_results=2)
                        if conv_res:
                            c_texts = [f"Past Discussion Summary: {c['summary']}" for c in conv_res if c["distance"] < 1.0]
                            if c_texts: v_ctx += "\n" + "\n".join(c_texts)
                except Exception: pass
                return v_ctx

            async def get_tools():
                now = time.time()
                if self._cached_info_tools and (now - self._last_tool_refresh <= self._refresh_interval):
                    return self._cached_info_tools
                
                logger.info("[ATLAS] Refreshing informational tool cache...")
                new_tools = []
                try:
                    status = mcp_manager.get_status()
                    # Subset of servers that Atlas can use independently for chat/research
                    configured_servers = set(mcp_manager.config.get("mcpServers", {}).keys())
                    discovery_servers = {"macos-use", "filesystem", "duckduckgo-search", "memory", "github", "weather", "search"}
                    
                    # Be proactive: try all discovery servers that are in the config, not just "connected" ones
                    active_servers = (configured_servers | {"filesystem", "memory"}) & discovery_servers
                    
                    logger.info(f"[ATLAS] Proactive tool discovery on servers: {active_servers}")
                    
                    # Parallel tool listing
                    server_tools = await asyncio.gather(*[mcp_manager.list_tools(s) for s in active_servers], return_exceptions=True)
                    
                    for s_name, t_list in zip(list(active_servers), server_tools):
                        if isinstance(t_list, (Exception, BaseException)): 
                            logger.warning(f"[ATLAS] Could not list tools for {s_name}: {t_list}")
                            continue
                        
                        # Explicitly cast to list to satisfy type checkers
                        for tool in cast(list, t_list):
                            t_low, d_low = tool.name.lower(), tool.description.lower()
                            # Broader 'safe' matching for solo research
                            is_safe = any(p in t_low or p in d_low for p in ["get", "list", "read", "search", "stats", "fetch", "check", "find", "view", "query", "cat", "ls"])
                            is_mut = any(p in t_low or p in d_low for p in ["create", "delete", "write", "update", "exec", "run", "set", "modify"])
                            
                            if is_safe and not is_mut:
                                new_tools.append({"name": f"{s_name}_{tool.name}", "description": tool.description, "input_schema": tool.inputSchema})

                    
                    self._cached_info_tools = new_tools
                    self._last_tool_refresh = int(now)
                    logger.info(f"[ATLAS] Cached {len(new_tools)} informational tools.")
                except Exception as e: logger.warning(f"[ATLAS] Tool discovery failed: {e}")
                return new_tools

            # Gather all context in parallel
            graph_context, vector_context, available_tools_info = await asyncio.gather(get_graph(), get_vector(), get_tools())


        # D. System Context (Always fast)
        try:
            ctx_snapshot = shared_context.to_dict()
            system_status = f"Project: {ctx_snapshot.get('project_root', 'Unknown')}\nVars: {ctx_snapshot.get('variables', {})}"
        except Exception:
            system_status = "Active."

        # E. DEEP THINKING: Trigger only for complex queries in the first turn
        analysis_context = ""
        is_complex = len(user_request.split()) > 7 or any(
            kw in user_request.lower()
            for kw in ["як", "чому", "виправ", "зроби", "поясни", "how", "why", "fix", "explain"]
        )

        if not is_simple_chat and is_complex:
            logger.info("[ATLAS] Engaging deep reasoning for chat...")
            reasoning = await self.use_sequential_thinking(user_request, total_thoughts=2)
            if reasoning.get("success"):
                analysis_context = f"\nDEEP ANALYSIS:\n{reasoning.get('analysis')}\n"

        # 2. Generate Super Prompt
        agent_capabilities = (
            "- Web search, File read, Spotlight, System info, GitHub/Docker info (Read-only)."
            if available_tools_info
            else "- Conversational assistant."
        )

        if intent == "solo_task":
            system_prompt_text = generate_atlas_solo_task_prompt(
                user_query=user_request,
                graph_context=graph_context,
                vector_context=vector_context,
                system_status=system_status,
                agent_capabilities=agent_capabilities,
                use_deep_persona=use_deep_persona,
            )
        else:
            system_prompt_text = generate_atlas_chat_prompt(
                user_query=user_request,
                graph_context=graph_context,
                vector_context=vector_context,
                system_status=system_status,
                agent_capabilities=agent_capabilities,
                use_deep_persona=use_deep_persona,
            )
        if analysis_context:
            system_prompt_text += f"\n{analysis_context}"

        from langchain_core.messages import BaseMessage

        messages: list[BaseMessage] = [SystemMessage(content=system_prompt_text)]
        if history:
            messages.extend(history[-10:])
        messages.append(HumanMessage(content=user_request))

        # 3. Tool Binding (Only if tools available)
        llm_instance = (
            self.llm.bind_tools(available_tools_info) if available_tools_info else self.llm
        )

        MAX_CHAT_TURNS = 5
        current_turn = 0

        while current_turn < MAX_CHAT_TURNS:
            response = await llm_instance.ainvoke(messages)

            if not getattr(response, "tool_calls", None):
                await self._memorize_chat_interaction(user_request, cast(str, response.content))
                return cast(str, response.content)

            # Process Tool Calls (Same logic as before but using cached info)
            for tool_call in response.tool_calls:
                logical_tool_name = tool_call.get("name")
                args = tool_call.get("args", {})

                if logical_tool_name:
                    logger.info(f"[ATLAS CHAT] Executing: {logical_tool_name}")
                    messages.append(response)
                    try:
                        # Use intelligent dispatch (handles server resolution & args)
                        result = await mcp_manager.dispatch_tool(logical_tool_name, args)
                        logger.info(f"[ATLAS CHAT] Tool result: {str(result)[:200]}...")
                    except Exception as tool_err:
                        logger.error(f"[ATLAS CHAT] Tool call failed: {tool_err}")
                        result = {"error": str(tool_err)}

                    from langchain_core.messages import ToolMessage

                    messages.append(
                        ToolMessage(
                            content=str(result)[:5000],
                            tool_call_id=tool_call.get("id", "chat_call"),
                        )
                    )
                else:
                    return cast(str, response.content)

            current_turn += 1

        fallback_msg = "Я виконав кілька кроків пошуку, але мені потрібно більше часу для повного аналізу. Що саме вас цікавить найбільше?"
        await self._memorize_chat_interaction(user_request, fallback_msg)
        return fallback_msg

    async def _memorize_chat_interaction(self, query: str, response: str):
        """Active memory consolidation for chat turns."""
        if not long_term_memory.available:
            return

        try:
            # Only memorize significant turns
            if len(query) > 5 or len(response) > 10:
                summary = f"User: {query}\nAtlas: {response[:300]}..."
                long_term_memory.remember_conversation(
                    session_id="chat_stream_global",
                    summary=summary,
                    metadata={"query_preview": query[:50], "timestamp": datetime.now().isoformat()},
                )
                logger.info("[ATLAS] Memorized chat interaction.")
        except Exception as e:
            logger.warning(f"[ATLAS] Memory write failed: {e}")

    async def create_plan(self, enriched_request: dict[str, Any]) -> TaskPlan:
        """
        Principal Architect: Creates an execution plan with Strategic Thinking.
        """
        import uuid

        from langchain_core.messages import HumanMessage, SystemMessage

        task_text = enriched_request.get("enriched_request", str(enriched_request))

        # 1. STRATEGIC ANALYSIS (Internal Thought)
        # complexity = enriched_request.get("complexity", "medium") # Removed to fix F841
        logger.info(f"[ATLAS] Deep Thinking: Analyzing strategy for '{task_text[:50]}...'")

        # Memory recall for strategy
        memory_context = ""
        if long_term_memory.available:
            similar = long_term_memory.recall_similar_tasks(task_text, n_results=2)
            if similar:
                memory_context = "\nPAST LESSONS (Strategies used before):\n" + "\n".join(
                    [f"- {s['document']}" for s in similar]
                )

            # --- BEHAVIORAL LEARNING RECALL ---
            behavioral_lessons = long_term_memory.recall_behavioral_logic(task_text, n_results=2)
            if behavioral_lessons:
                memory_context += "\n\nPAST BEHAVIORAL DEVIATIONS (LEARNED LOGIC):\n" + "\n".join(
                    [f"- {b['document']}" for b in behavioral_lessons]
                )
                logger.info(
                    f"[ATLAS] Recalled {len(behavioral_lessons)} behavioral lessons for planning."
                )

        simulation_prompt = AgentPrompts.atlas_simulation_prompt(task_text, memory_context)

        try:
            sim_resp = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a Strategic Architect. Think technically and deeply in English."
                    ),
                    HumanMessage(content=simulation_prompt),
                ]
            )
            simulation_result = cast(
                str, sim_resp.content if hasattr(sim_resp, "content") else str(sim_resp)
            )
        except Exception as e:
            logger.warning(f"[ATLAS] Deep Thinking failed: {e}")
            simulation_result = "Standard execution strategy."

        # 2. PLAN FORMULATION
        intent = enriched_request.get("intent", "task")

        # Inject context-specific doctrine
        if intent == "development":
            doctrine = AgentPrompts.SDLC_PROTOCOL
        else:
            doctrine = AgentPrompts.TASK_PROTOCOL

        dynamic_system_prompt = self.SYSTEM_PROMPT.replace(
            "{{CONTEXT_SPECIFIC_DOCTRINE}}", doctrine
        )

        # 2.5 MCP INFRASTRUCTURE CONTEXT (For adaptive step assignment)
        mcp_context = await self._get_mcp_capabilities_context()
        active_servers = mcp_context.get("active_servers", [])
        mcp_context_str = f"""
AVAILABLE MCP INFRASTRUCTURE (DYNAMICALLY DETERMINED):
Active Servers: {', '.join([s['name'] for s in active_servers])}

Server Details (sorted by priority):
{chr(10).join([f"- {s['name']} (Tier {s['tier']}): {s['description']} | Connected: {s['connected']}" for s in active_servers[:8]])}

CRITICAL PLANNING RULES:
1. ONLY assign steps to servers that are ACTIVE (listed above)
2. Prefer Tier 1 servers (macos-use, filesystem) for core operations
3. Use server's 'when_to_use' guidance when choosing tools
4. For web tasks: prefer puppeteer/duckduckgo-search over macos-use browser
5. Each step MUST specify 'realm' (server name) for Tetyana
"""

        prompt = AgentPrompts.atlas_plan_creation_prompt(
            task_text,
            simulation_result,
            (
                shared_context.available_mcp_catalog
                if hasattr(shared_context, "available_mcp_catalog")
                else ""
            ),
            "",  # vibe_directive handled via SDLC_PROTOCOL inside doctrine
            str(shared_context.to_dict()),
        )

        # Append MCP infrastructure context for adaptive planning
        prompt += f"\n\n{mcp_context_str}"

        messages = [
            SystemMessage(content=dynamic_system_prompt),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        plan_data = self._parse_response(cast(str, response.content))

        # ENSURE VOICE_ACTION INTEGRITY: Post-process steps to guarantee Ukrainian descriptions
        steps = plan_data.get("steps", [])
        import re

        for step in steps:
            # If voice_action is missing or contains English, force a generic Ukrainian description
            va = step.get("voice_action", "")
            if not va or re.search(r"[a-zA-Z]", va):
                # Fallback heuristic: Try to translate action intent
                action = step.get("action", "").lower()
                if "click" in action:
                    va = "Виконую натискання на елемент"
                elif "type" in action:
                    va = "Вводжу текст"
                elif "search" in action:
                    va = "Шукаю інформацію"
                elif "vibe" in action:
                    va = "Запускаю аналіз Вайб"
                elif "terminal" in action or "command" in action:
                    va = "Виконую команду в терміналі"
                else:
                    va = "Переходжу до наступного етапу завдання"
                step["voice_action"] = va
                logger.warning(f"[ATLAS] Fixed missing/English voice_action in step: {va}")

        # META-PLANNING FALLBACK: If planner failed to generate steps, force reasoning
        if not steps:
            logger.info(
                "[ATLAS] No direct steps found. Engaging Meta-Planning via sequential-thinking..."
            )
            reasoning = await self.use_sequential_thinking(task_text)
            if reasoning.get("success"):
                # Re-try planning with reasoning context
                prompt += f"\n\nRESEARCH FINDINGS:\n{reasoning.get('analysis')!s}"
                messages = [
                    SystemMessage(content=dynamic_system_prompt),
                    HumanMessage(content=prompt),
                ]
                response = await self.llm.ainvoke(messages)
                plan_data = self._parse_response(cast(str, response.content))
                steps = plan_data.get("steps", [])
                # Re-check voice_action for new steps
                for step in steps:
                    if not step.get("voice_action") or re.search(
                        r"[a-zA-Z]", step.get("voice_action", "")
                    ):
                        step["voice_action"] = "Виконую заплановану дію"

        self.current_plan = TaskPlan(
            id=str(uuid.uuid4())[:8],
            goal=str(plan_data.get("goal", enriched_request.get("enriched_request", ""))),
            steps=steps,
            context={**enriched_request, "simulation": simulation_result},
        )

        return self.current_plan

    async def get_grisha_report(self, step_id: str) -> str | None:
        """Retrieve Grisha's detailed rejection report from notes or memory"""
        import ast
        import json
        import os

        from ..mcp_manager import mcp_manager

        def _parse_payload(payload: Any) -> dict[str, Any] | None:
            if isinstance(payload, dict):
                return payload
            if hasattr(payload, "structuredContent") and isinstance(
                payload.structuredContent, dict
            ):
                return payload.structuredContent.get("result", payload.structuredContent)
            if hasattr(payload, "content"):
                for item in getattr(payload, "content", []) or []:
                    text = getattr(item, "text", None)
                    if isinstance(text, dict):
                        return text
                    if isinstance(text, str):
                        try:
                            return json.loads(text)
                        except Exception:
                            try:
                                return ast.literal_eval(text)
                            except Exception:
                                continue
            return None

        # Try filesystem first (faster and cleaner)
        try:
            reports_dir = os.path.expanduser("~/.config/atlastrinity/reports")
            if os.path.exists(reports_dir):
                # Find reports for this step
                candidates = [
                    f
                    for f in os.listdir(reports_dir)
                    if f.startswith(f"rejection_step_{step_id}_") and f.endswith(".md")
                ]

                if candidates:
                    # Sort by timestamp (part of filename) descending
                    candidates.sort(reverse=True)
                    latest_report = os.path.join(reports_dir, candidates[0])

                    with open(latest_report, encoding="utf-8") as f:
                        content = f.read()

                    logger.info(
                        f"[ATLAS] Retrieved Grisha's report from filesystem: {latest_report}"
                    )
                    return content
        except Exception as e:
            logger.warning(f"[ATLAS] Could not retrieve from filesystem: {e}")

        # Fallback to memory
        try:
            result = await mcp_manager.call_tool(
                "memory", "search_nodes", {"query": f"grisha_rejection_step_{step_id}"}
            )

            if result and hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        return item.text
            elif isinstance(result, dict) and "entities" in result:
                entities = result["entities"]
                if entities and len(entities) > 0:
                    return entities[0].get("observations", [""])[0]

            logger.info(f"[ATLAS] Retrieved Grisha's report from memory for step {step_id}")
        except Exception as e:
            logger.warning(f"[ATLAS] Could not retrieve from memory: {e}")

        return None

    async def help_tetyana(self, step_id: str, error: str) -> dict[str, Any]:
        """Helps Tetyana when she is stuck, using shared context and Grisha's feedback for better solutions"""
        from langchain_core.messages import HumanMessage, SystemMessage

        # Get context for better recovery suggestions
        context_info = shared_context.to_dict()

        # Try to get Grisha's detailed report
        grisha_report = await self.get_grisha_report(step_id)
        grisha_feedback = ""
        if grisha_report:
            grisha_feedback = f"\n\nGRISHA'S DETAILED FEEDBACK:\n{grisha_report}\n"

        prompt = AgentPrompts.atlas_help_tetyana_prompt(
            int(step_id)
            if isinstance(step_id, str) and step_id.isdigit()
            else (int(step_id) if isinstance(step_id, int | float) else 0),
            error,
            grisha_feedback,
            context_info,
            self.current_plan.steps if self.current_plan else [],
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        logger.info(f"[ATLAS] Helping Tetyana with context: {context_info}")
        response = await self.llm.ainvoke(messages)
        return self._parse_response(cast(str, response.content))

    async def evaluate_healing_strategy(
        self, error: str, vibe_report: str, grisha_audit: dict, context: dict | None = None
    ) -> dict[str, Any]:
        """
        Atlas reviews the diagnostics from Vibe and the audit from Grisha.
        Decides whether to proceed with the self-healing fix and sets the tempo.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        context_data = context or shared_context.to_dict()

        prompt = AgentPrompts.atlas_healing_review_prompt(
            error, vibe_report, grisha_audit, context_data
        )

        messages = [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=prompt)]

        try:
            logger.info("[ATLAS] Reviewing self-healing strategy and setting tempo...")
            response = await self.llm.ainvoke(messages)
            decision = self._parse_response(cast(str, response.content))

            logger.info(f"[ATLAS] Healing Decision: {decision.get('decision', 'PIVOT')}")
            return decision
        except Exception as e:
            logger.error(f"[ATLAS] Healing review failed: {e}")
            return {
                "decision": "PIVOT",
                "reason": f"Review failed due to technical error: {e!s}",
                "voice_message": "Я не зміг узгодити план лікування. Спробую інший підхід.",
            }

    async def summarize_session(self, messages: list[Any]) -> dict[str, Any]:
        """Generate a professional summary and extract key entities from a session."""
        if not messages:
            return {"summary": "Empty session", "entities": []}

        # Format conversation for LLM
        conv_text = ""
        for msg in messages[-50:]:  # Take last 50 messages for summary
            role = "USER" if "HumanMessage" in str(type(msg)) else "ATLAS"
            content = msg.content if hasattr(msg, "content") else str(msg)
            conv_text += f"{role}: {content[:500]}\n"

        prompt = f"""Analyze the following conversation and provide:
        1. A professional, detailed technical summary in ENGLISH (max 500 chars).
        2. A list of key entities, names, or concepts mentioned (max 10).

        CONVERSATION:
        {conv_text}

        Respond in JSON:
        {{
            "summary": "...",
            "entities": ["name1", "concept2", ...]
        }}
        """

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content="You are a Professional Archivist."),
                    HumanMessage(content=prompt),
                ]
            )
            content = cast(str, response.content if hasattr(response, "content") else str(response))

            # JSON extraction
            import json

            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"Failed to summarize session: {e}")
            return {"summary": "Summary failed", "entities": []}

    async def evaluate_execution(self, goal: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Atlas reviews the execution results of Tetyana and Grisha.
        Determines if the goal was REALLY achieved and if the strategy is worth remembering.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Prepare execution summary for LLM
        history = ""
        for i, res in enumerate(results):
            status = "✅" if res.get("success") else "❌"
            history += f"{i + 1}. [{res.get('step_id')}] {res.get('action')}: {status} {str(res.get('result'))[:2000]}\n"
            if res.get("error"):
                history += f"   Error: {res.get('error')}\n"

        logger.info(f"[ATLAS] Deep Evaluating execution quality for goal: {goal[:50]}...")

        # 1. Deep Reasoning Phase (Fact Extraction)
        reasoning_query = f"""Analyze this execution history and extract precise facts to answer the user's goal.
GOAL: {goal}
HISTORY: {history}

Extract specific numbers, names, and technical outcomes. If the user asked to count, find the count in the results.
Output internal thoughts in English, then prepare a final report in UKRAINIAN with 0% English words."""

        try:
            # Use Sequential Thinking for analysis
            analysis_result = await self.use_sequential_thinking(reasoning_query)
            synthesis_context = str(analysis_result.get("analysis", "No deep analysis available."))
        except Exception as e:
            logger.warning(f"Sequential thinking for evaluation failed: {e}")
            synthesis_context = "Fallback to direct synthesis."

        # 2. Final Synthesis Phase (JSON Formatting)
        prompt = f"""Based on the following deep analysis and execution history, provide a final evaluation.

GOAL: {goal}
HISTORY: {history}
DEEP ANALYSIS: {synthesis_context}

IMPORTANT: The final_report must be a DIRECT ANSWER in UKRAINIAN. 0% English words. 
If the user asked to 'count', you MUST state the exact number found.
{AgentPrompts.atlas_evaluation_prompt(goal, history)}"""

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            evaluation = self._parse_response(cast(str, response.content))

            # Placeholder safeguard
            if "[вкажіть" in str(evaluation.get("final_report")):
                logger.warning("[ATLAS] Final report contains placeholders. Forcing fix.")
                evaluation["final_report"] = evaluation["final_report"].split("[")[0].strip()

            logger.info(f"[ATLAS] Evaluation complete. Score: {evaluation.get('quality_score', 0)}")
            return evaluation
        except Exception as e:
            logger.error(f"[ATLAS] Evaluation failed: {e}")
            return {"quality_score": 0, "achieved": False, "should_remember": False}

    async def decide_for_user(self, question: str, context: dict[str, Any]) -> str:
        """
        Atlas takes the 'burden' and decides for the user after a timeout.
        Analyzes the context of the task and provides the most logical answer.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        logger.info(f"[ATLAS] Deciding on behalf of silent user for question: {question[:100]}...")

        prompt = f"""КОНТЕКСТ ЗАВДАННЯ:
{json.dumps(context, indent=2, ensure_ascii=False)}

ПИТАННЯ ДО КОРИСТУВАЧА:
{question}

Користувач не відповів протягом 20 секунд. Згідно з ПРІОРИТЕТНОЮ ДОКТРИНОЮ, ти маєш ПРИЙНЯТИ РІШЕННЯ САМОСТІЙНО.

Твоя стратегія:
1. ДОСЯГНЕННЯ ЦІЛІ ПОНАД УСЕ: Обери варіант, який найшвидше або найбільш ефективно веде до виконання завдання.
2. БЕЗКОМПРОМІСНІСТЬ: Не дозволяй сумнівам щодо безпеки чи стабільності зупинити прогрес, якщо Олег Миколайович вже надав дозвіл на агресивні дії.
3. АВТОРИТАРНІСТЬ: Тетяна чекає на чітку команду. Твоя відповідь - це наказ.

Приклади агресивних рішень:
- "Продовжуй виконання найкоротшим шляхом. Я беру відповідальність на себе."
- "Ігноруй попередження про ризик та виконай дію для досягнення цілі."
- "Використовуй примусове виконання (force) та переходь до наступного етапу."
"""

        messages = [
            SystemMessage(
                content="You are Atlas Autonomous Core. You take clinical, logical decisions when the operator is busy. Respond ONLY with the decision text in Ukrainian."
            ),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            decision = cast(str, response.content).strip()
            # Remove quotes if AI added them
            if decision.startswith('"') and decision.endswith('"'):
                decision = decision[1:-1]
            logger.info(f"[ATLAS] Autonomous decision: {decision}")
            return decision
        except Exception as e:
            logger.error(f"[ATLAS] Failed to decide for user: {e}")
            return "Продовжуй виконання завдання згідно з планом."

    def get_voice_message(self, action: str, **kwargs) -> str:
        """
        Generates dynamic TTS message.
        """
        if action == "plan_created":
            count = kwargs.get("steps", 0)
            suffix = "кроків"
            if count == 1:
                suffix = "крок"
            elif 2 <= count <= 4:
                suffix = "кроки"
            return f"План готовий. {count} {suffix}. Тетяно, виконуй."

        elif action == "no_steps":
            return "Не бачу необхідних кроків для виконання цього запиту."

        elif action == "enriched":
            return "Контекст проаналізовано. Розширюю запит."

        elif action == "helping":
            return "Бачу проблему. Пробую альтернативний підхід."

        elif action == "delegating":
            return "Тетяно, передаю керування тобі."

        elif action == "recovery_started":
            # Atlas should not claim the step 'stopped' (validation is Grisha's duty).
            return f"Розпочинаю допомогу у відновленні кроку {kwargs.get('step_id', '?')} — консультація та аналіз."

        elif action == "vibe_engaged":
            return (
                f"Залучаю Вайб для глибинного аналізу помилки у кроці {kwargs.get('step_id', '?')}."
            )

        return f"Атлас: {action}"
