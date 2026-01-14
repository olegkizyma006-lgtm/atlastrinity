from .common import DEFAULT_REALM_CATALOG  # re-export default catalog
from .atlas import ATLAS
from .tetyana import TETYANA
from .grisha import GRISHA
from ..config import WORKSPACE_DIR

__all__ = ["DEFAULT_REALM_CATALOG", "ATLAS", "TETYANA", "GRISHA", "AgentPrompts"]


class AgentPrompts:
    """Compatibility wrapper that exposes the same interface while sourcing prompts from modular files"""

    ATLAS = ATLAS
    TETYANA = TETYANA
    GRISHA = GRISHA

    @staticmethod
    def tetyana_reasoning_prompt(
        step: str, context: dict, tools_summary: str = "", feedback: str = ""
    ) -> str:
        feedback_section = f"\n        PREVIOUS REJECTION FEEDBACK (from Grisha):\n        {feedback}\n" if feedback else ""
        
        return f"""Analyze how to execute this atomic step: {step}.

        CONTEXT: {context}
        {feedback_section}
        {tools_summary}

        Your task is to choose the BEST tool and arguments.
        CRITICAL: Follow the 'Schema' provided for each tool EXACTLY. Ensure parameter names and types match.
        If there is feedback from Grisha above, ADAPT your strategy to address his concerns.

        Respond in JSON:
        {{
            "thought": "Internal technical analysis in ENGLISH (Which tool? Which args? Why based on schema?)",
            "proposed_action": {{ "tool": "name", "args": {{...}} }},
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
        step_action: str, expected_result: str, context: dict, overall_goal: str = ""
    ) -> str:
        return f"""You are the Verification Strategist.

        OVERALL GOAL: {overall_goal}
        Step: {step_action}
        Expected Result: {expected_result}

        Create a verification strategy:
        1. VISION FIRST: What should I see on the screenshot to confirm success?
        2. TOOL FALLBACK: If the screen is not enough, what MCP tool (ls, grep, etc.) should I use?

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
        overall_goal: str = "",
    ) -> str:
        return f"""Verify the result of the following step using MCP tools FIRST, screenshots only when necessary.

    OVERALL GOAL: {overall_goal}
    STRATEGIC GUIDANCE (Follow this!):
    {strategy_context}

    Step {step_id}: {step_action}
    Expected Result: {expected}
    Actual Output/Result: {actual}

    Shared Context (for correct paths and global situation): {context_info}

    Verification History (Tool actions taken during this verification): {history}

    PRIORITY ORDER FOR VERIFICATION:
    1. Use MCP tools to verify results (filesystem, terminal, git, etc.)
    2. Check files, directories, command outputs directly
    3. ONLY use screenshots for visual/UI verification when explicitly needed

    Analyze the current situation. If you can verify using MCP tools, do that first.
    Use 'macos-use.screenshot' ONLY when you need to verify visual elements, UI, or when explicitly mentioned in expected result.

    CRITICAL: When rejecting a result (verified: false), you MUST provide:
    1. "description": Detailed technical explanation in English (what went wrong, what was expected vs what happened)
    2. "issues": Array of specific problems found (e.g., ["File not created", "Command failed", "Wrong directory structure"])
    3. "voice_message": Clear Ukrainian message for Atlas and Tetyana explaining the rejection
    4. "confidence": Your confidence level (0.0-1.0)
    5. "remediation_suggestions": Array of actionable fixes (e.g., ["Create missing file", "Run command with correct flags"])"""

    # --- ATLAS PROMPTS ---

    @staticmethod
    def atlas_intent_classification_prompt(
        user_request: str, context: str, history: str
    ) -> str:
        return f"""Analyze the user request and decide if it's a simple conversation, a technical task, or a SOFTWARE DEVELOPMENT task.

User Request: {user_request}
Context: {context}
Conversation History: {history}

CRITICAL CLASSIFICATION RULES:
1. 'chat' - Greetings, 'How are you', jokes, appreciation (thanks), or SIMPLE CONFIRMATIONS.
2. 'task' - Direct instructions to DO something (open app, run command, search file).
3. 'development' - Requests to CREATE, BUILD, or WRITE software, code, scripts, apps, websites, APIs.
   Examples: "Create a Python script", "Build a website", "Write an API", "Develop a bot"

If request is 'development', set complexity to 'high' and use_vibe to true.

ALL textual responses (reason, initial_response) MUST be in UKRAINIAN.

Respond STRICTLY in JSON:
{{
    "intent": "chat" or "task" or "development",
    "reason": "Explain your choice in Ukrainian",
    "enriched_request": "Detailed description of the request (English)",
    "complexity": "low/medium/high",
    "use_vibe": true/false (true for development tasks),
    "initial_response": "Short reply to user ONLY if intent is 'chat' (Ukrainian), else null"
}}
"""

    @staticmethod
    def atlas_chat_prompt() -> str:
        return """You are in friendly conversation mode.
Your role: Witty, smart interlocutor Atlas.
Style: Concise, with humor.
LANGUAGE: You MUST respond in UKRAINIAN only!
Do not suggest creating a plan, just talk."""

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
    ) -> str:
        return f"""Create a Master Execution Plan.

        REQUEST: {task_text}
        STRATEGY: {strategy}
        {vibe_directive}
        {catalog}

        CONSTRAINTS:
        - Output JSON matching the format in your SYSTEM PROMPT.
        - 'goal', 'reason', and 'action' descriptions MUST be in English (technical precision).
        - 'voice_summary' MUST be in UKRAINIAN (for the user).
        - **NO META-STEPS**: Skip steps like "Think about X", "Classify Y", or "Verify Z". Only plan DIRECT tasks.

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
    def grisha_strategist_system_prompt(decision_context: str) -> str:
        return f"""You are a Verification Strategist. Consider the environment and choose the best verification stack.
        ENVIRONMENT_DECISION: {decision_context}
        When visual evidence is conclusive, prioritize Vision verification. When authoritative system/data checks are needed, prefer MCP servers (favor local Swift-based MCP servers when available). Output internal strategies in English."""
