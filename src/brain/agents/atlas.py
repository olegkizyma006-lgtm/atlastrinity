"""
Atlas - The Strategist

Role: Strategic analysis, plan formulation, task delegation
Voice: Dmytro (male)
Model: GPT-4.1 / GPT-5 mini
"""

import os
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
# Check root (Dev: src/brain/agents -> root)
root_dev = os.path.join(current_dir, "..", "..", "..")
# Check resources (Prod: brain/agents -> Resources)
root_prod = os.path.join(current_dir, "..", "..")

for r in [root_dev, root_prod]:
    abs_r = os.path.abspath(r)
    if abs_r not in sys.path:
        sys.path.insert(0, abs_r)

from providers.copilot import CopilotLLM  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from ..config_loader import config  # noqa: E402
from ..context import shared_context  # noqa: E402
from ..logger import logger  # noqa: E402
from ..memory import long_term_memory  # noqa: E402
from ..prompts import AgentPrompts  # noqa: E402
from ..prompts.atlas_chat import generate_atlas_chat_prompt  # noqa: E402
from .base_agent import BaseAgent  # noqa: E402


@dataclass
class TaskPlan:
    """Execution plan structure"""

    id: str
    goal: str
    steps: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, active, completed, failed
    context: Dict[str, Any] = field(default_factory=dict)


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

    def __init__(self, model_name: str = "raptor-mini"):
        # Get model config (config.yaml > parameter > env variables)
        agent_config = config.get_agent_config("atlas")
        final_model = model_name
        if model_name == "raptor-mini":  # default parameter
            final_model = agent_config.get("model") or os.getenv("COPILOT_MODEL", "raptor-mini")

        self.llm = CopilotLLM(model_name=final_model)
        
        # Optimization: Tool Cache
        self._cached_info_tools = []
        self._last_tool_refresh = 0
        self._refresh_interval = 1800 # 30 minutes
        self.temperature = agent_config.get("temperature", 0.7)
        self.current_plan: Optional[TaskPlan] = None
        self.history: List[Dict[str, Any]] = []

    async def use_sequential_thinking(
        self, problem: str, available_tools: list = None
    ) -> Dict[str, Any]:
        """
        Use sequential-thinking MCP for deep reasoning on complex problems.
        Returns structured analysis with step-by-step recommendations.
        """
        from ..logger import logger  # noqa: E402
        from ..mcp_manager import mcp_manager  # noqa: E402

        if available_tools is None:
            available_tools = [
                "terminal",
                "filesystem",
                "browser",
                "gui",
                "applescript",
            ]

        try:
            result = await mcp_manager.call_tool(
                "sequential-thinking",
                "sequentialthinking_tools",
                {
                    "available_mcp_tools": available_tools,
                    "thought": f"Analyzing task: {problem}",
                    "thought_number": 1,
                    "total_thoughts": 5,
                    "next_thought_needed": True,
                    "current_step": {
                        "step_description": "Initial analysis",
                        "expected_outcome": "Clear understanding of the problem",
                        "recommended_tools": [],
                    },
                },
            )
            logger.info(f"[ATLAS] Sequential thinking result: {str(result)[:300]}")
            return {"success": True, "analysis": result}
        except Exception as e:
            logger.warning(f"[ATLAS] Sequential thinking unavailable: {e}")
            return {"success": False, "error": str(e)}

    async def analyze_request(
        self,
        user_request: str,
        context: Dict[str, Any] = None,
        history: List[Any] = None,
    ) -> Dict[str, Any]:
        """Analyzes user request: determines intent (chat vs task)"""

        req_lower = user_request.lower().strip()

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
            analysis = self._parse_response(response.content)
            
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
                "initial_response": None # Force falling back to atlas.chat() for dynamic response
            }

    async def evaluate_deviation(self, current_step: dict, proposed_deviation: str, full_plan: list) -> dict:
        """
        Evaluates a strategic deviation proposed by Tetyana.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        prompt = AgentPrompts.atlas_deviation_evaluation_prompt(
            str(current_step),
            proposed_deviation,
            context=shared_context.to_dict(),
            full_plan=str(full_plan)
        )
        
        # Strip system prompt placeholder
        system_prompt = self.SYSTEM_PROMPT.replace("{{CONTEXT_SPECIFIC_DOCTRINE}}", "")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            evaluation = self._parse_response(response.content)
            logger.info(f"[ATLAS] Deviation Evaluation: {evaluation.get('approved')}")
            return evaluation
        except Exception as e:
            logger.error(f"[ATLAS] Evaluation failed: {e}")
            return {"approved": False, "reason": "Evaluation failed", "voice_message": "Помилка оцінки."}
    
    async def chat(self, user_request: str, history: List[Any] = None, use_deep_persona: bool = False) -> str:
        """
        Omni-Knowledge Chat Mode.
        Integrates Graph Memory, Vector Memory, and System Context for deep awareness.
        """
        import time
        from langchain_core.messages import HumanMessage, SystemMessage
        from ..mcp_manager import mcp_manager

        # 1. Fast-Path: Simple Greeting Detection
        # If it's a short greeting, skip expensive lookups and tool discovery
        is_simple_chat = len(user_request.split()) < 4 and any(g in user_request.lower() for g in ["привіт", "хай", "hello", "hi", "атлас", "atlas", "як справи", "що ти"])
        
        graph_context = ""
        vector_context = ""
        system_status = ""
        available_tools_info = []

        if not is_simple_chat:
            # A. Graph Memory (MCP Search) - ONLY FOR COMPLEX QUERIES
            try:
                # Use faster text-only lookup for chat context
                graph_res = await mcp_manager.call_tool("memory", "search_nodes", {"query": user_request})
                if isinstance(graph_res, dict) and "entities" in graph_res:
                    entities = graph_res.get("results", []) # Corrected from 'entities' to 'results' based on memory_server.py
                    if entities:
                        graph_chunks = [f"Entity: {e.get('name')} | Info: {'; '.join(e.get('observations', [])[:2])}" for e in entities[:2]]
                        graph_context = "\n".join(graph_chunks)
            except Exception:
                pass

            # B. Vector Memory (ChromaDB)
            try:
                if long_term_memory.available:
                    # Search similar tasks/plans
                    similar_tasks = long_term_memory.recall_similar_tasks(user_request, n_results=1)
                    if similar_tasks:
                        vector_context += "\nPast Strategy: " + similar_tasks[0]['document'][:200]
                    
                    # Search past conversations!
                    similar_convs = long_term_memory.recall_similar_conversations(user_request, n_results=2)
                    if similar_convs:
                        conv_texts = [f"Past Discussion Summary: {c['summary']}" for c in similar_convs if c['distance'] < 1.0]
                        if conv_texts:
                            vector_context += "\n" + "\n".join(conv_texts)
            except Exception:
                pass

            # Update Atlas history window from 10 to 25 (handled in Orchestrator call, but we note it here)
            # history = history[-25:] if len(history) > 25 else history

            # C. Dynamic Tool Discovery (With Caching & Safe Spawn)
            now = time.time()
            if not self._cached_info_tools or (now - self._last_tool_refresh > self._refresh_interval):
                logger.info("[ATLAS] Refreshing informational tool cache...")
                new_tools = []
                try:
                    mcp_config = mcp_manager.config.get("mcpServers", {})
                    # Only fetch tools from servers that are already CONNECTED or ACTIVE
                    # to avoid massive posix_spawn bursts for cold servers
                    status = mcp_manager.get_status()
                    active_servers = set(status.get("connected_servers", [])) | {"macos-use", "filesystem", "duckduckgo-search", "memory"}
                    
                    for server_name in active_servers:
                        if server_name not in mcp_config: continue
                        try:
                            tools_list = await mcp_manager.list_tools(server_name)
                            for tool in tools_list:
                                t_lower = tool.name.lower()
                                d_lower = tool.description.lower()
                                
                                is_safe = any(p in t_lower or p in d_lower for p in ["get", "list", "read", "search", "stats", "status", "fetch", "explain", "check", "verify", "info", "find", "show", "view", "query"])
                                is_mutative = any(p in t_lower or p in d_lower for p in ["create", "delete", "write", "update", "move", "remove", "kill", "stop", "start", "exec", "run", "set", "change", "modify", "clear"])
                                
                                if is_safe and not is_mutative:
                                    new_tools.append({
                                        "name": f"{server_name}_{tool.name}",
                                        "description": tool.description,
                                        "input_schema": tool.inputSchema
                                    })
                        except Exception:
                            continue # Skip servers that fail to list tools to avoid blocking
                            
                    self._cached_info_tools = new_tools
                    self._last_tool_refresh = now
                    logger.info(f"[ATLAS] Cached {len(new_tools)} informational tools.")
                except Exception as e:
                    logger.warning(f"[ATLAS] Tool discovery throttled: {e}")
            
            available_tools_info = self._cached_info_tools

        # D. System Context (Always fast)
        try:
            ctx_snapshot = shared_context.to_dict()
            system_status = f"Project: {ctx_snapshot.get('project_root', 'Unknown')}\nVars: {ctx_snapshot.get('variables', {})}"
        except Exception:
            system_status = "Active."

        # 2. Generate Super Prompt
        agent_capabilities = "- Web search, File read, Spotlight, System info, GitHub/Docker info (Read-only)." if available_tools_info else "- Conversational assistant."
        
        system_prompt_text = generate_atlas_chat_prompt(
            user_query=user_request,
            graph_context=graph_context,
            vector_context=vector_context,
            system_status=system_status,
            agent_capabilities=agent_capabilities,
            use_deep_persona=use_deep_persona,
        )

        messages = [SystemMessage(content=system_prompt_text)]
        if history: messages.extend(history[-10:])
        messages.append(HumanMessage(content=user_request))

        # 3. Tool Binding (Only if tools available)
        llm_instance = self.llm.bind_tools(available_tools_info) if available_tools_info else self.llm
        
        MAX_CHAT_TURNS = 5
        current_turn = 0
        
        while current_turn < MAX_CHAT_TURNS:
            response = await llm_instance.ainvoke(messages)
            
            if not getattr(response, "tool_calls", None):
                return response.content
            
            # Process Tool Calls (Same logic as before but using cached info)
            for tool_call in response.tool_calls:
                logical_tool_name = tool_call.get("name")
                args = tool_call.get("args", {})
                
                if "_" in logical_tool_name:
                    parts = logical_tool_name.split("_", 1)
                    mcp_server, mcp_tool = parts[0], parts[1]
                else:
                    mcp_server, mcp_tool = "", logical_tool_name
                
                if mcp_server:
                    logger.info(f"[ATLAS CHAT] Executing: {mcp_server}:{mcp_tool}")
                    messages.append(response)
                    try:
                        result = await mcp_manager.call_tool(mcp_server, mcp_tool, args)
                        logger.info(f"[ATLAS CHAT] Tool result: {str(result)[:200]}...")
                    except Exception as tool_err:
                        logger.error(f"[ATLAS CHAT] Tool call failed: {tool_err}")
                        result = {"error": str(tool_err)}
                    
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(result)[:5000],
                        tool_call_id=tool_call.get("id", "chat_call")
                    ))
                else:
                    return response.content
            
            current_turn += 1

        return "Я виконав кілька кроків пошуку, але мені потрібно більше часу для повного аналізу. Що саме вас цікавить найбільше?"

    async def create_plan(self, enriched_request: Dict[str, Any]) -> TaskPlan:
        """
        Principal Architect: Creates an execution plan with Strategic Thinking.
        """
        import uuid  # noqa: E402

        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

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
                logger.info(f"[ATLAS] Recalled {len(behavioral_lessons)} behavioral lessons for planning.")

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
            simulation_result = sim_resp.content if hasattr(sim_resp, "content") else str(sim_resp)
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
            
        dynamic_system_prompt = self.SYSTEM_PROMPT.replace("{{CONTEXT_SPECIFIC_DOCTRINE}}", doctrine)

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

        messages = [
            SystemMessage(content=dynamic_system_prompt),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        plan_data = self._parse_response(response.content)

        # ENSURE VOICE_ACTION INTEGRITY: Post-process steps to guarantee Ukrainian descriptions
        steps = plan_data.get("steps", [])
        import re
        for step in steps:
            # If voice_action is missing or contains English, force a generic Ukrainian description
            va = step.get("voice_action", "")
            if not va or re.search(r'[a-zA-Z]', va):
                # Fallback heuristic: Try to translate action intent
                action = step.get("action", "").lower()
                if "click" in action: va = "Виконую натискання на елемент"
                elif "type" in action: va = "Вводжу текст"
                elif "search" in action: va = "Шукаю інформацію"
                elif "vibe" in action: va = "Запускаю аналіз Вайб"
                elif "terminal" in action or "command" in action: va = "Виконую команду в терміналі"
                else: va = "Переходжу до наступного етапу завдання"
                step["voice_action"] = va
                logger.warning(f"[ATLAS] Fixed missing/English voice_action in step: {va}")

        # META-PLANNING FALLBACK: If planner failed to generate steps, force reasoning
        if not steps:
            logger.info("[ATLAS] No direct steps found. Engaging Meta-Planning via sequential-thinking...")
            reasoning = await self.use_sequential_thinking(task_text)
            if reasoning.get("success"):
                # Re-try planning with reasoning context
                prompt += f"\n\nRESEARCH FINDINGS:\n{str(reasoning.get('analysis'))}"
                messages = [
                    SystemMessage(content=dynamic_system_prompt),
                    HumanMessage(content=prompt),
                ]
                response = await self.llm.ainvoke(messages)
                plan_data = self._parse_response(response.content)
                steps = plan_data.get("steps", [])
                # Re-check voice_action for new steps
                for step in steps:
                    if not step.get("voice_action") or re.search(r'[a-zA-Z]', step.get("voice_action", "")):
                        step["voice_action"] = "Виконую заплановану дію"

        self.current_plan = TaskPlan(
            id=str(uuid.uuid4())[:8],
            goal=plan_data.get("goal", enriched_request.get("enriched_request", "")),
            steps=steps,
            context={**enriched_request, "simulation": simulation_result},
        )

        return self.current_plan

    async def use_sequential_thinking(self, task: str) -> Dict[str, Any]:
        """
        Engage sequential-thinking MCP server for deep reasoning and meta-planning.
        """
        from ..mcp_manager import mcp_manager  # noqa: E402
        
        logger.info(f"[ATLAS] Deep Reasoning for: {task[:50]}...")
        
        prompt = f"""You are Atlas's Reasoning Core. Analyze this task and suggest 3-5 concrete technical steps.
        TASK: {task}
        
        USE 'sequentialthinking' tool to brainstorm and provide a structured analysis.
        """
        
        try:
            # Using unified dispatch_tool
            res = await mcp_manager.dispatch_tool(
                "sequential-thinking",
                {
                    "thought": f"Initial analysis of goal: {task}",
                    "thoughtNumber": 1,
                    "totalThoughts": 3,
                    "nextThoughtNeeded": True
                }
            )
            
            # Follow up with more details
            res2 = await mcp_manager.dispatch_tool(
                "sequential-thinking",
                {
                    "thought": f"Exploring technical barriers and alternatives for {task}...",
                    "thoughtNumber": 2,
                    "totalThoughts": 3,
                    "nextThoughtNeeded": True
                }
            )
            
            # Final synthesis
            res3 = await mcp_manager.call_tool(
                "sequential-thinking",
                "sequentialthinking",
                {
                    "thought": f"Final strategy formulated for {task}.",
                    "thoughtNumber": 3,
                    "totalThoughts": 3,
                    "nextThoughtNeeded": False
                }
            )
            
            # Helper to extract text from result
            def _get_text(r):
                if isinstance(r, dict):
                    return str(r.get("content", r.get("error", "")))
                if hasattr(r, "content") and isinstance(r.content, list):
                    return "".join([getattr(item, "text", "") for item in r.content if hasattr(item, "text")])
                return str(r)

            # Combine thoughts into a 'result'
            analysis = (
                f"THOUGHT 1: {_get_text(res)}\n"
                f"THOUGHT 2: {_get_text(res2)}\n"
                f"THOUGHT 3: {_get_text(res3)}"
            )
            
            return {"success": True, "analysis": analysis}
        except Exception as e:
            logger.error(f"[ATLAS] Sequential thinking failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_grisha_report(self, step_id: int) -> Optional[str]:
        """Retrieve Grisha's detailed rejection report from notes or memory"""
        from ..mcp_manager import mcp_manager  # noqa: E402

        import ast  # noqa: E402
        import json  # noqa: E402
        import os  # noqa: E402

        def _parse_payload(payload: Any) -> Optional[Dict[str, Any]]:
            if isinstance(payload, dict):
                return payload
            if hasattr(payload, "structuredContent") and isinstance(
                getattr(payload, "structuredContent"), dict
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
                candidates = [f for f in os.listdir(reports_dir) if f.startswith(f"rejection_step_{step_id}_") and f.endswith(".md")]
                
                if candidates:
                    # Sort by timestamp (part of filename) descending
                    candidates.sort(reverse=True)
                    latest_report = os.path.join(reports_dir, candidates[0])
                    
                    with open(latest_report, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    logger.info(f"[ATLAS] Retrieved Grisha's report from filesystem: {latest_report}")
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

    async def help_tetyana(self, step_id: int, error: str) -> Dict[str, Any]:
        """Helps Tetyana when she is stuck, using shared context and Grisha's feedback for better solutions"""
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

        # Get context for better recovery suggestions
        context_info = shared_context.to_dict()

        # Try to get Grisha's detailed report
        grisha_report = await self.get_grisha_report(step_id)
        grisha_feedback = ""
        if grisha_report:
            grisha_feedback = f"\n\nGRISHA'S DETAILED FEEDBACK:\n{grisha_report}\n"

        prompt = AgentPrompts.atlas_help_tetyana_prompt(
            step_id,
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
        return self._parse_response(response.content)

    async def summarize_session(self, messages: List[Any]) -> Dict[str, Any]:
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
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a Professional Archivist."),
                HumanMessage(content=prompt)
            ])
            content = response.content if hasattr(response, "content") else str(response)
            
            # JSON extraction
            import json
            start = content.find("{")
            end = content.rfind("}") + 1
            return json.loads(content[start:end])
        except Exception as e:
            logger.error(f"Failed to summarize session: {e}")
            return {"summary": "Summary failed", "entities": []}

    async def evaluate_execution(self, goal: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Atlas reviews the execution results of Tetyana and Grisha.
        Determines if the goal was REALLY achieved and if the strategy is worth remembering.
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

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
            response = await self.llm.ainvoke([
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            evaluation = self._parse_response(response.content)
            
            # Placeholder safeguard
            if "[вкажіть" in str(evaluation.get("final_report")):
                logger.warning("[ATLAS] Final report contains placeholders. Forcing fix.")
                evaluation["final_report"] = evaluation["final_report"].split("[")[0].strip()

            logger.info(f"[ATLAS] Evaluation complete. Score: {evaluation.get('quality_score', 0)}")
            return evaluation
        except Exception as e:
            logger.error(f"[ATLAS] Evaluation failed: {e}")
            return {"quality_score": 0, "achieved": False, "should_remember": False}

    async def decide_for_user(self, question: str, context: Dict[str, Any]) -> str:
        """
        Atlas takes the 'burden' and decides for the user after a timeout.
        Analyzes the context of the task and provides the most logical answer.
        """
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
        
        logger.info(f"[ATLAS] Deciding on behalf of silent user for question: {question[:100]}...")
        
        prompt = f"""КОНТЕКСТ ЗАВДАННЯ:
{json.dumps(context, indent=2, ensure_ascii=False)}

ПИТАННЯ ДО КОРИСТУВАЧА:
{question}

Користувач не відповів вчасно АБО логіка виконання вимагає прийняття рішення для продовження.
Твоє завдання: ПРИЙНЯТИ ОСТАТОЧНЕ РІШЕННЯ замість користувача.

Твоя відповідь буде передана Тетяні як пряма вказівка до дії.
Пиши УКРАЇНСЬКОЮ мовою. Будь конкретним та авторитарним у рішенні.

ПРИКЛАДИ: 
- "Продовжуй виконання кроку, я підтверджую дію."
- "Пропусти цей крок та переходь до наступного, це не критично."
- "Використовуй останній відомий шлях, не чекай відповіді."
"""
        
        messages = [
            SystemMessage(content="You are Atlas Autonomous Core. You take clinical, logical decisions when the operator is busy. Respond ONLY with the decision text in Ukrainian."),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            decision = response.content.strip()
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
            return f"Крок {kwargs.get('step_id', '?')} зупинився. Шукаю рішення."

        elif action == "vibe_engaged":
            return (
                f"Залучаю Вайб для глибинного аналізу помилки у кроці {kwargs.get('step_id', '?')}."
            )

        return f"Атлас: {action}"
