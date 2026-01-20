"""
BaseAgent - Shared utilities for Trinity agents

This module provides common functionality used by Atlas, Tetyana, and Grisha agents.
"""

import json
from typing import Any


class BaseAgent:
    """Base class for Trinity agents with shared utilities."""

    def _parse_response(self, content: str) -> dict[str, Any]:
        """
        Parse JSON response from LLM with fuzzy fallback.

        Handles:
        1. Clean JSON responses
        2. JSON embedded in text
        3. YAML-like key:value pairs
        4. Raw text fallback
        """
        # 1. Try standard JSON extraction
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass

        # 2. Fuzzy YAML-like parsing (handles LLM responses like "verified: true\nconfidence: 0.9")
        try:
            data = {}
            for line in content.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    # Handle boolean values
                    if value.lower() == "true":
                        data[key] = True
                    elif value.lower() == "false":
                        data[key] = False
                    # Handle numeric values
                    elif value.replace(".", "", 1).isdigit():
                        data[key] = float(value)
                    else:
                        data[key] = value

            # Consider it valid fuzzy parse if we found key fields
            if "verified" in data or "intent" in data or "success" in data:
                return data
        except Exception:
            pass

        # 3. Return raw content as fallback
        return {"raw": content}

    async def use_sequential_thinking(self, task: str, total_thoughts: int = 3) -> dict[str, Any]:
        """
        Universal reasoning capability for any agent.
        Uses a dedicated LLM (as configured in sequential_thinking.model) to generate
        deep thoughts and stores them via the sequential-thinking MCP tool.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from providers.copilot import CopilotLLM

        from ..config_loader import config
        from ..logger import logger
        from ..mcp_manager import mcp_manager

        agent_name = self.__class__.__name__.upper()
        logger.info(f"[{agent_name}] 🤔 Thinking deeply about: {task[:60]}...")

        # 1. Get model from config (defaulting to raptor-mini as requested)
        seq_config = config.get("mcp.sequential_thinking", {})
        model_name = seq_config.get("model", "raptor-mini")

        # 2. Initialize dedicated thinker
        # We need to ensure providers is in path, usually it's there via agent init overrides
        try:
            thinker_llm = CopilotLLM(model_name=model_name)
        except ImportError:
            logger.error("Could not import CopilotLLM. Ensure 'providers' is in sys.path")
            return {"success": False, "analysis": "Reflexion failed due to import error"}

        full_analysis = ""
        current_context = ""  # Accumulate thoughts for LLM context

        try:
            for i in range(1, total_thoughts + 1):
                is_last = i == total_thoughts

                # 3. Ask LLM for the next thought
                prompt = f"""You are a deep technical reasoning engine.
TASK: {task}

PREVIOUS THOUGHTS:
{current_context}

STEP {i}/{total_thoughts}:
Generate the next logical thought to analyze this problem. 
- Focus on root causes, technical details, and specific actionable solutions.
- If this is the final thought, provide a summary and recommendation.
- Output ONLY the raw thought text. Do not wrap in JSON or Markdown blocks.
"""
                response = await thinker_llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are a Sequential Thinking Engine. Output ONLY the raw thought text."
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                thought_content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # 4. Record thought via MCP tool (records history)
                logger.debug(f"[{agent_name}] Thought cycle {i}: {thought_content[:100]}...")

                await mcp_manager.dispatch_tool(
                    "sequential-thinking.sequentialthinking",
                    {
                        "thought": thought_content,
                        "thoughtNumber": i,
                        "totalThoughts": total_thoughts,
                        "nextThoughtNeeded": not is_last,
                    },
                )

                # Update context for next iteration
                current_context += f"Thought {i}: {thought_content}\n"
                full_analysis += f"\n[Thought {i}]: {thought_content[:800]}..."

            logger.info(f"[{agent_name}] Reasoning complete using model {model_name}.")
            return {"success": True, "analysis": full_analysis}

        except Exception as e:
            logger.warning(f"[{agent_name}] Sequential thinking unavailable/failed: {e}")
            # Do not crash the agent, just return failure so it can fallback to standard logic
            return {"success": False, "error": str(e), "analysis": full_analysis}
