"""
Atlas - The Strategist

Role: Strategic analysis, plan formulation, task delegation
Voice: Dmytro (male)
Model: GPT-4.1 / GPT-5 mini
"""

import os
# Import provider
# Robust path handling for both Dev and Production (Packaged)
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

from ..config_loader import config  # noqa: E402
from ..context import shared_context  # noqa: E402
from ..logger import logger  # noqa: E402
from ..memory import long_term_memory  # noqa: E402
from ..prompts import AgentPrompts  # noqa: E402
from ..prompts.atlas_chat import generate_atlas_chat_prompt  # noqa: E402


@dataclass
class TaskPlan:
    """Execution plan structure"""

    id: str
    goal: str
    steps: List[Dict[str, Any]]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, active, completed, failed
    context: Dict[str, Any] = field(default_factory=dict)


class Atlas:
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
            final_model = agent_config.get("model") or os.getenv(
                "COPILOT_MODEL", "raptor-mini"
            )

        self.llm = CopilotLLM(model_name=final_model)
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
        from ..logger import logger
        from ..mcp_manager import mcp_manager

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
        from langchain_core.messages import HumanMessage, SystemMessage

        req_lower = user_request.lower().strip()

        # Comprehensive layout-agnostic and conversational heuristic
        # logger.info(f"Chat keywords: {chat_keywords}") # Removed to fix F841

        # Optimized conversational detection - stricter rules to avoid false positives with tasks
        import re

        # Regex for word boundaries to avoid substring matches (e.g., "hi" in "history")
        def has_word(text, word):
            return re.search(r"\b" + re.escape(word) + r"\b", text) is not None

        # Greetings
        greeting_words = ["привіт", "здоров", "вітаю", "ghbdsn", "ghbdtn"]
        is_greeting = (
            any(w in req_lower for w in greeting_words)
            or has_word(req_lower, "hi")
            or has_word(req_lower, "hello")
        )

        # How are you (Strict phrases only)
        # Avoid checking single particles like "ся" or "ти" which appear in normal text
        how_are_you_phrases = [
            "як справи",
            "як ти",
            "як ви",
            "як воно",
            "як ся маєш",
            "що нового",
            "rfr ltkf",
        ]
        is_how_are_you = any(phrase in req_lower for phrase in how_are_you_phrases)

        # Confirmations (Simple positive responses)
        confirm_words = ["так", "звісно", "ага", "ок", "добре", "зрозумів", "fuf", "lf"]
        is_confirm = any(req_lower == w for w in confirm_words)

        # Thanks
        thanks_words = ["дякую", "спасибі", "дкю", "dfre.", "cgfcb,"]
        is_thanks = any(w in req_lower for w in thanks_words) or has_word(
            req_lower, "thanks"
        )

        # Check for simple greetings or casual match
        if (
            is_greeting
            or is_how_are_you
            or is_thanks
            or is_confirm
            or len(req_lower) < 2
        ):
            response_type = "greeting"
            if is_how_are_you:
                response_type = "how_are_you"

            initial_response = "Привіт! Я на зв'язку."
            if response_type == "how_are_you":
                initial_response = "Привіт! У мене все чудово. Чим можу допомогти?"
            elif req_lower in ["так", "звісно", "ага", "ок", "добре"]:
                initial_response = "Чудово! Продовжуємо."

            return {
                "intent": "chat",
                "reason": "Розпізнано розмовний характер, привітання або просте підтвердження.",
                "enriched_request": user_request,
                "complexity": "low",
                "initial_response": initial_response,
            }

        # Check for software development keywords
        dev_keywords_ua = ["напиши код", "створи програму", "розроби", "напрограмуй", "скрипт", "веб-сайт", "апі", "бекенд", "фронтенд", "додаток"]
        dev_keywords_en = ["create app", "build", "develop", "code", "implement", "write script", "api", "website", "frontend", "backend", "program", "software"]
        
        is_development = (
            any(kw in req_lower for kw in dev_keywords_ua) or
            any(kw in req_lower for kw in dev_keywords_en)
        )

        prompt = AgentPrompts.atlas_intent_classification_prompt(
            user_request, str(context or "None"), str(history or "None")
        )
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_response(response.content)
        except Exception as e:
            # If LLM fails (e.g. 400 error), default to chat for safety unless it looks very much like a command
            logger.error(f"Intent detection LLM failed: {e}")
            is_likely_task = any(
                cmd in req_lower
                for cmd in ["відкрий", "запусти", "terminal", "python", "file", "код"]
            )
            return {
                "intent": "task" if is_likely_task else "chat",
                "reason": f"Помилка API ({e}). Автоматичне визначення.",
                "enriched_request": user_request,
                "complexity": "low",
                "initial_response": (
                    "Я трохи підвисаю, але готовий спілкуватися."
                    if not is_likely_task
                    else None
                ),
            }

    async def chat(self, user_request: str, history: List[Any] = None) -> str:
        """
        Omni-Knowledge Chat Mode.
        Integrates Graph Memory, Vector Memory, and System Context for deep awareness.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from ..mcp_manager import mcp_manager

        # 1. Gather Context from Memory Arsenal
        graph_context = ""
        vector_context = ""
        
        # A. Graph Memory (MCP Search)
        try:
            # Search for relevant entities in the Knowledge Graph
            graph_res = await mcp_manager.call_tool(
                "memory", "search_nodes", {"query": user_request}
            )
            
            # Format graph result
            if isinstance(graph_res, dict) and "entities" in graph_res:
                entities = graph_res.get("entities", [])
                if entities:
                    graph_chunks = []
                    for e in entities[:3]: # Top 3 entities
                        name = e.get("name", "Unknown")
                        obs = "; ".join(e.get("observations", [])[:2])
                        graph_chunks.append(f"Entity: {name} | Info: {obs}")
                    graph_context = "\n".join(graph_chunks)
            elif hasattr(graph_res, "content"):
                # Handle FastMCP text content
                content_list = getattr(graph_res, "content", [])
                text_content = [getattr(c, "text", "") for c in content_list if hasattr(c, "text")]
                graph_context = "\n".join(text_content)[:800]
                
        except Exception as e:
            logger.warning(f"[ATLAS] Chat Memory lookup failed: {e}")

        # B. Vector Memory (ChromaDB)
        try:
            if long_term_memory.available:
                # Look for similar past successful tasks/conversations
                similar_tasks = long_term_memory.recall_similar_tasks(user_request, n_results=2)
                if similar_tasks:
                    vector_context += "\nRelated Tasks:\n" + "\n".join([f"- {s['document'][:200]}..." for s in similar_tasks])
                
                # Look for lessons learned (errors) relevant to this topic
                similar_errors = long_term_memory.recall_similar_errors(user_request, n_results=1)
                if similar_errors:
                    vector_context += "\nRelated Lessons:\n" + "\n".join([f"- {s['document'][:200]}..." for s in similar_errors])
        except Exception as e:
            logger.warning(f"[ATLAS] Vector Memory lookup failed: {e}")

        # C. System Status
        try:
            # Snapshot of shared context
            ctx_snapshot = shared_context.to_dict()
            system_status = f"Current Project: {ctx_snapshot.get('project_root', 'Unknown')}\n"
            system_status += f"Variables: {ctx_snapshot.get('variables', {})}"
        except Exception:
            system_status = "System running."

        # D. Agent Capabilities info
        agent_capabilities = """
        - TETYANA: Executor (Terminal, Filesystem, Browser). Contact for ACTIONS.
        - GRISHA: Verifier (Vision, Screenshots, Security). Contact for AUDIT.
        - VIBE: Developer (Coding, Debugging, Self-Healing). Contact for CODE.
        """

        # 2. Generate Super Prompt
        system_prompt_text = generate_atlas_chat_prompt(
            user_query=user_request,
            graph_context=graph_context,
            vector_context=vector_context,
            system_status=system_status,
            agent_capabilities=agent_capabilities
        )

        # 3. Invoke LLM
        messages = [SystemMessage(content=system_prompt_text)]
        
        # Add historical context (last 10 messages)
        if history:
            messages.extend(history[-10:])
        
        # Add current user message
        messages.append(HumanMessage(content=user_request))

        logger.info(f"[ATLAS] Generating chat response with full memory context...")
        response = await self.llm.ainvoke(messages)
        
        if hasattr(response, "content"):
            return response.content
        return str(response)

    async def create_plan(self, enriched_request: Dict[str, Any]) -> TaskPlan:
        """
        Principal Architect: Creates an execution plan with Strategic Thinking.
        """
        import uuid

        from langchain_core.messages import HumanMessage, SystemMessage

        task_text = enriched_request.get("enriched_request", str(enriched_request))

        # 1. STRATEGIC ANALYSIS (Internal Thought)
        # complexity = enriched_request.get("complexity", "medium") # Removed to fix F841
        logger.info(
            f"[ATLAS] Deep Thinking: Analyzing strategy for '{task_text[:50]}...'"
        )

        # Memory recall for strategy
        memory_context = ""
        if long_term_memory.available:
            similar = long_term_memory.recall_similar_tasks(task_text, n_results=2)
            if similar:
                memory_context = (
                    "\nPAST LESSONS (Strategies used before):\n"
                    + "\n".join([f"- {s['document']}" for s in similar])
                )

        simulation_prompt = AgentPrompts.atlas_simulation_prompt(
            task_text, memory_context
        )

        try:
            sim_resp = await self.llm.ainvoke(
                [
                    SystemMessage(
                        content="You are a Strategic Architect. Think technically and deeply in English."
                    ),
                    HumanMessage(content=simulation_prompt),
                ]
            )
            simulation_result = (
                sim_resp.content if hasattr(sim_resp, "content") else str(sim_resp)
            )
        except Exception as e:
            logger.warning(f"[ATLAS] Deep Thinking failed: {e}")
            simulation_result = "Standard execution strategy."

        # 2. PLAN FORMULATION
        use_vibe = enriched_request.get("use_vibe", False) or enriched_request.get("intent") == "development"
        vibe_directive = ""
        
        if use_vibe:
            vibe_directive = """
            CRITICAL DEVELOPMENT OVERRIDE:
            This is a SOFTWARE DEVELOPMENT task. You MUST delegate to 'vibe' server (Mistral AI) for:
            1. Planning architecture (vibe_smart_plan)
            2. Writing code (vibe_prompt)
            3. Debugging (vibe_analyze_error)
            
            Do NOT attempt to write complex code yourself via filesystem. Delegate to Vibe.
            """

        prompt = AgentPrompts.atlas_plan_creation_prompt(
            task_text,
            simulation_result,
            (
                shared_context.available_mcp_catalog
                if hasattr(shared_context, "available_mcp_catalog")
                else ""
            ),
            vibe_directive,
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = await self.llm.ainvoke(messages)
        plan_data = self._parse_response(response.content)

        self.current_plan = TaskPlan(
            id=str(uuid.uuid4())[:8],
            goal=plan_data.get("goal", enriched_request.get("enriched_request", "")),
            steps=plan_data.get("steps", []),
            context={**enriched_request, "simulation": simulation_result},
        )

        return self.current_plan

    async def get_grisha_report(self, step_id: int) -> Optional[str]:
        """Retrieve Grisha's detailed rejection report from notes or memory"""
        from ..mcp_manager import mcp_manager

        # Try notes first (faster and cleaner)
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

            if result and hasattr(result, "content"):
                for item in result.content:
                    if hasattr(item, "text"):
                        data = (
                            eval(item.text) if isinstance(item.text, str) else item.text
                        )
                        if data.get("notes") and len(data["notes"]) > 0:
                            note_id = data["notes"][0]["id"]
                            # Read full note
                            note_result = await mcp_manager.call_tool(
                                "notes", "read_note", {"note_id": note_id}
                            )
                            if note_result and hasattr(note_result, "content"):
                                for note_item in note_result.content:
                                    if hasattr(note_item, "text"):
                                        note_data = (
                                            eval(note_item.text)
                                            if isinstance(note_item.text, str)
                                            else note_item.text
                                        )
                                        logger.info(
                                            f"[ATLAS] Retrieved Grisha's report from notes for step {step_id}"
                                        )
                                        return note_data.get("content", "")
            elif (
                isinstance(result, dict)
                and result.get("success")
                and result.get("notes")
            ):
                notes = result["notes"]
                if notes and len(notes) > 0:
                    note_id = notes[0]["id"]
                    note_result = await mcp_manager.call_tool(
                        "notes", "read_note", {"note_id": note_id}
                    )
                    if isinstance(note_result, dict) and note_result.get("success"):
                        logger.info(
                            f"[ATLAS] Retrieved Grisha's report from notes for step {step_id}"
                        )
                        return note_result.get("content", "")
        except Exception as e:
            logger.warning(f"[ATLAS] Could not retrieve from notes: {e}")

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

            logger.info(
                f"[ATLAS] Retrieved Grisha's report from memory for step {step_id}"
            )
        except Exception as e:
            logger.warning(f"[ATLAS] Could not retrieve from memory: {e}")

        return None

    async def help_tetyana(self, step_id: int, error: str) -> Dict[str, Any]:
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

    async def evaluate_execution(
        self, goal: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Atlas reviews the execution results of Tetyana and Grisha.
        Determines if the goal was REALLY achieved and if the strategy is worth remembering.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Prepare execution summary for LLM
        history = ""
        for i, res in enumerate(results):
            status = "✅" if res.get("success") else "❌"
            history += f"{i + 1}. [{res.get('step_id')}] {res.get('action')}: {status} {str(res.get('result'))[:200]}\n"
            if res.get("error"):
                history += f"   Error: {res.get('error')}\n"

        prompt = AgentPrompts.atlas_evaluation_prompt(goal, history)

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        logger.info(f"[ATLAS] Evaluating execution quality for goal: {goal[:50]}...")
        try:
            response = await self.llm.ainvoke(messages)
            evaluation = self._parse_response(response.content)
            logger.info(
                f"[ATLAS] Evaluation complete. Score: {evaluation.get('quality_score', 0)}"
            )
            return evaluation
        except Exception as e:
            logger.error(f"[ATLAS] Evaluation failed: {e}")
            return {"quality_score": 0, "achieved": False, "should_remember": False}

    def get_voice_message(self, action: str, **kwargs) -> str:
        """
        Generates dynamic TTS message.
        """
        if action == "plan_created":
            count = kwargs.get('steps', 0)
            suffix = "кроків"
            if count == 1: suffix = "крок"
            elif 2 <= count <= 4: suffix = "кроки"
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
            return f"Залучаю Vibe для глибинного аналізу помилки у кроці {kwargs.get('step_id', '?')}."

        return f"Атлас: {action}"

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        import json

        try:
            # Find JSON in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        return {"raw": content}
