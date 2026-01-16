from ..config import WORKSPACE_DIR
from .atlas import ATLAS
from .common import DEFAULT_REALM_CATALOG  # re-export default catalog
from .grisha import GRISHA
from .tetyana import TETYANA

__all__ = ["DEFAULT_REALM_CATALOG", "ATLAS", "TETYANA", "GRISHA", "AgentPrompts"]


class AgentPrompts:
    """Compatibility wrapper that exposes the same interface while sourcing prompts from modular files"""

    ATLAS = ATLAS
    TETYANA = TETYANA
    GRISHA = GRISHA

    @staticmethod
    def tetyana_reasoning_prompt(
        step: str,
        context: dict,
        tools_summary: str = "",
        feedback: str = "",
        previous_results: list = None,
        goal_context: str = "",
        bus_messages: list = None,
        full_plan: str = "",
    ) -> str:
        feedback_section = (
            f"\n        PREVIOUS REJECTION FEEDBACK (from Grisha):\n        {feedback}\n"
            if feedback
            else ""
        )

        results_section = ""
        if previous_results:
            # Format results nicely
            formatted_results = []
            for res in previous_results:
                # Truncate long outputs
                res_str = str(res)
                if len(res_str) > 1000:
                    res_str = res_str[:1000] + "...(truncated)"
                formatted_results.append(res_str)
            results_section = f"\n        RESULTS OF PREVIOUS STEPS (Use this data to fill arguments):\n        {formatted_results}\n"

        plan_section = f"\n        FULL MASTER EXECUTION PLAN (Follow this sequence strictly):\n        {full_plan}\n" if full_plan else ""

        return f"""Analyze how to execute this atomic step: {step}.
        {goal_section}
        {plan_section}
        CONTEXT: {context}
        {results_section}
        {feedback_section}
        {bus_section}
        {tools_summary}

        Your task is to choose the BEST tool and arguments.
        CRITICAL RULES:
        1. Follow the 'Schema' provided for each tool EXACTLY.
        2. ADHERE STRICTLY to the plan sequence above. Do not skip or reorder steps.
        3. If there is feedback from Grisha or other agents above, ADAPT your strategy to address their concerns.
        4. If you are unsure or need clarification from Atlas to proceed, use the "question_to_atlas" field.

        Respond in JSON:
        {{
            "thought": "Internal technical analysis in ENGLISH (Which tool? Which args? Why based on schema?)",
            "proposed_action": {{ "tool": "name", "args": {{...}} }},
            "question_to_atlas": "Optional technical question if you are stuck or need guidance",
            "voice_message": "Ukrainian message for the user describing the action"
        }}
        """

    @staticmethod
    def tetyana_reflexion_prompt(
        step: str, error: str, history: list, tools_summary: str = ""
    ) -> str:
        return f"""Analysis of Failure: {error}.

        Step: {step}
        History of attempts: {history}
        {tools_summary}

        Determine if you can fix this by changing the TOOL or ARGUMENTS for THIS step.
        If the failure is logical or requires changing the goal, set "requires_atlas": true.

        Respond in JSON:
        {{
            "analysis": "Technical cause of failure (English)",
            "fix_attempt": {{ "tool": "name", "args": {{...}} }},
            "requires_atlas": true/false,
            "question_to_atlas": "Optional technical question if you need Atlas's specific help",
            "voice_message": "Ukrainian explanation of why it failed and how you are fixing it"
        }}
        """

    @staticmethod
    def tetyana_execution_prompt(step: str, context_results: list) -> str:
        return f"""Execute this task step: {step}.
    Current context results: {context_results}
    Respond ONLY with JSON:
    {{
        "analysis": "Ukrainian explanation",
        "tool_call": {{ "name": "...", "args": {{...}} }},
        "voice_message": "Ukrainian message for user"
    }}
    """

    @staticmethod
    def grisha_strategy_prompt(
        step_action: str, expected_result: str, context: dict, goal_context: str = ""
    ) -> str:
        return f"""You are the Verification Strategist. 
        Your task is to create a robust verification plan for the following step:
        
        {goal_context}
        Step: {step_action}
        Expected Result: {expected_result}

        Design a strategy using the available environment resources. 
        Choose whether to use Vision (screenshots/OCR) or MCP Tools (system data/files) or BOTH.
        Prefer high-precision native tools for data and Vision for visual state.
        
        CRITICAL: Focus ONLY on proving that THIS specific step succeeded as expected.
        Do not demand the entire goal to be finished if this is just one step in a sequence.

        Strategy:
        """

    @staticmethod
    def grisha_verification_prompt(
        strategy_context: str,
        step_id: int,
        step_action: str,
        expected: str,
        actual: str,
        context_info: dict,
        history: list,
        goal_context: str = "",
        tetyana_thought: str = "",
    ) -> str:
        return f"""Verify the result of the following step using MCP tools FIRST, screenshots only when necessary.

    OVERALL CONTEXT:
    {goal_context}
    
    STRATEGIC GUIDANCE (Follow this!):
    {strategy_context}

    Step {step_id}: {step_action}
    Expected Result: {expected}
    Actual Output/Result: {actual}
    
    TETYANA'S INTENT (Monologue from execution):
    {tetyana_thought or "No thought documented."}

    Shared Context (for correct paths and global situation): {context_info}

    Verification History (Tool actions taken during this verification): {history}

    PRIORITY ORDER FOR VERIFICATION:
    1. Use MCP tools to verify results (filesystem, terminal, git, etc.)
    2. Check files, directories, command outputs directly
    3. ONLY use screenshots for visual/UI verification when explicitly needed

    Analyze the current situation. If you can verify using MCP tools, do that first.
    Use 'macos-use_take_screenshot' for visual UI verification.
    Use 'macos-use_analyze_screen' for screen text (OCR) analysis.
    
    CRITICAL VERIFICATION RULE:
    - You are verifying STEP {step_id}: "{step_action}".
    - If the "Actual Output" or Tool Results prove that THIS step's "Expected Result" is met, then VERIFIED=TRUE.
    - Do NOT reject the result because the overall task/goal is not yet finished. You are only auditor for this atomic step.

    TRUST THE TOOLS:
    - If an MCP tool returns a success result (process ID, file content, search results), ACCEPT IT.
    - REASONING TOOLS: If 'sequential-thinking' or 'vibe_ask' provides a thought process or analysis, TRUST IT as proof of execution for logic-based steps.
    - Do NOT reject technical success just because you didn't see it visually on a screenshot.
    - If the goal was to kill a process and 'pgrep' returns nothing, that is SUCCESS.

    Respond STRICTLY in JSON.
    
    Example SUCCESS response:
    {{
      "action": "verdict",
      "verified": true,
      "confidence": 1.0,
      "description": "Terminal output confirms file was created successfully.",
      "voice_message": "Завдання виконано."
    }}

    Example REJECTION response:
    {{
      "action": "verdict",
      "verified": false,
      "confidence": 0.8,
      "description": "Expected to find directory 'mac-discovery' with specific structure, but directory does not exist.",
      "issues": ["Directory 'mac-discovery' not found"],
      "voice_message": "Результат не прийнято. Директорія не створена.",
      "remediation_suggestions": ["Create mac-discovery directory"]
    }}"""

    # --- ATLAS PROMPTS ---

    @staticmethod
    def atlas_intent_classification_prompt(user_request: str, context: str, history: str) -> str:
        return f"""Analyze the user request and decide if it's a simple conversation, a technical task, or a SOFTWARE DEVELOPMENT task.

User Request: {user_request}
Context: {context}
Conversation History: {history}

CRITICAL CLASSIFICATION RULES:
1. 'chat' - Greetings, appreciation, jokes, or INFORMATION-SEEKING questions (weather, explanations of scripts, general info, GitHub searches) that do NOT require modifying the system or creating files.
2. 'task' - Direct instructions to DO something (open app, run command, move file, system control).
3. 'development' - Requests to CREATE, BUILD, or WRITE software, complex code, scripts, apps, websites, APIs.
   Examples: "Create a Python script", "Build a website", "Write an API"

DEEP PERSONA TRIGGER:
If the user wants to talk about YOUR identity, purpose, philosophy, the program's soul, existence, our shared history, or "heart-to-heart" topics, set 'use_deep_persona' to true.

If request is 'development' or a high-complexity 'task', set use_vibe to true.
If the user asks a question like "How does this script work?" or "Find me some interesting GitHub projects", CLASSIFY AS 'chat'.

ALL textual responses (reason) MUST be in UKRAINIAN.

Respond STRICTLY in JSON:
{{
    "intent": "chat" or "task" or "development",
    "reason": "Explain your choice in Ukrainian",
    "enriched_request": "Detailed description of the request (English)",
    "complexity": "low/medium/high",
    "use_vibe": true/false,
    "use_deep_persona": true/false
}}
"""

    @staticmethod
    def atlas_chat_prompt() -> str:
        return f"""You are in CAPABLE conversation mode.
Your role: Witty, smart, and HIGHLY INFORMED interlocutor Atlas.
Style: Concise, witty, but technical if needed.
LANGUAGE: You MUST respond in UKRAINIAN only!

CAPABILITIES:
- You have access to TOOLS (Search, Web Fetch, Knowledge Graph, Sequential Thinking).
- USE THEM for factual accuracy (weather, news, script explanation, GitHub research).
- If the user asks a question you don't know the answer to, SEARCH for it.
- Mental reasoning (thoughts) should be in English.

Do not suggest creating a complex plan, just use your tools autonomously to answer the user's question directly in chat."""

    @staticmethod
    def atlas_simulation_prompt(task_text: str, memory_context: str) -> str:
        return f"""Think deeply as a Strategic Architect about: {task_text}
        {memory_context}

        Analyze:
        1. Underlying logic of the task.
        2. Sequence of apps/tools needed.
        3. Potential technical barriers on macOS.

        Respond in English with a technical strategy.
        """

    @staticmethod
    def atlas_plan_creation_prompt(
        task_text: str,
        strategy: str,
        catalog: str,
        vibe_directive: str = "",
        context: str = "",
    ) -> str:
        context_section = f"\n        ENVIRONMENT & PATHS:\n        {context}\n" if context else ""
        
        return f"""Create a Master Execution Plan.

        REQUEST: {task_text}
        STRATEGY: {strategy}
        {context_section}
        {vibe_directive}
        {catalog}

        CONSTRAINTS:
        - Output JSON matching the format in your SYSTEM PROMPT.
        - 'goal', 'reason', and 'action' descriptions MUST be in English (technical precision).
        - 'voice_summary' MUST be in UKRAINIAN (for the user).
        - **META-PLANNING AUTHORIZED**: If the task is complex, you MAY include reasoning steps (using `sequential-thinking`) to discover the path forward. Do not just say "no steps found". Goal achievement is mandatory.

        Steps should be atomic and logical.
        """

    @staticmethod
    def atlas_help_tetyana_prompt(
        step_id: int,
        error: str,
        grisha_feedback: str,
        context_info: dict,
        current_plan: list,
    ) -> str:
        return f"""Tetyana is stuck at step {step_id}.

 Error: {error}
 {grisha_feedback}

 SHARED CONTEXT: {context_info}

 Current plan: {current_plan}

 You are the Meta-Planner. Provide an ALTERNATIVE strategy or a structural correction.
 IMPORTANT: If Grisha provided detailed feedback above, use it to understand EXACTLY what went wrong and avoid repeating the same mistake.

 Output JSON matching the 'help_tetyana' schema:
 {{
     "reason": "English analysis of the failure (incorporate Grisha's feedback if available)",
     "alternative_steps": [
         {{"id": 1, "action": "English description", "expected_result": "English description"}}
     ],
     "voice_message": "Short Ukrainian message explaining the pivot to the user"
 }}
 """

    @staticmethod
    def atlas_evaluation_prompt(goal: str, history: str) -> str:
        return f"""Review the execution of the following task.

        GOAL: {goal}

        EXECUTION HISTORY:
        {history}

        CRITICAL EVALUATION:
        1. Did we achieve the actual goal?
        2. Was the path efficient?
        3. Is this a 'Golden Path' that should be a lesson for the future?

        Respond STRICTLY in JSON:
        {{
            "quality_score": 0.0 to 1.0 (float),
            "achieved": true/false,
            "analysis": "Critique in UKRAINIAN",
            "compressed_strategy": [
                "Step 1 intent",
                "Step 2 intent",
                ...
            ],
            "should_remember": true/false
        }}
        """

    # --- GRISHA PROMPTS ---

    @staticmethod
    def grisha_security_prompt(action_str: str) -> str:
        return f"""Analyze this action for security risks: {action_str}

        Risks to check:
        1. Data loss (deletion, overwrite)
        2. System damage (system files, configs)
        3. Privacy leaks (uploading keys, passwords)

        Respond in JSON:
        {{
            "safe": true/false,
            "risk_level": "low/medium/high/critical",
            "reason": "English technical explanation",
            "requires_confirmation": true/false,
            "voice_message": "Ukrainian warning if risky, else empty"
        }}
        """

    @staticmethod
    def grisha_strategist_system_prompt(env_info: str) -> str:
        return f"""You are a Verification Strategist. 
Your goal is to decide the best way to verify a step outcome: Vision Framework vs MCP Tools.

AVAILABLE ENVIRONMENT INFO:
{env_info}

GUIDELINES:
- If the result is visual (UI layout, widget state, visual artifacts), prioritize 'macos-use_take_screenshot' and Vision analyze.
- If the result is system-level (files, processes, database, git), prioritize MCP tools (filesystem, terminal, etc.).
- Favor 'macos-use' for everything related to macOS interface and system control.
- You can combine tools if needed for multi-layer verification.
- Be precise and efficient. Do not request screenshots if a simple 'ls' or 'pgrep' provides the proof.

Output your internal verification strategy in English. Do NOT use markdown formatting for the strategy itself, just plain text."""
