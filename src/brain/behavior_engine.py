"""Behavior Engine - Config-Driven Decision System

Centralizes all behavioral logic through YAML configuration.
Replaces hardcoded conditionals with dynamic pattern matching.

Previously scattered across:
- adaptive_behavior.py: Behavior patterns and strategy selection
- atlas.py: Intent classification and greeting detection
- tool_dispatcher.py: Tool routing and synonym mapping
- mcp_registry.py: Task classification
- orchestrator.py: State machine decisions
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from .logger import logger


@dataclass
class Pattern:
    """Represents a behavior pattern from config."""

    name: str
    description: str
    trigger: dict[str, Any]
    action: dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0


class RuleEvaluator(Protocol):
    """Protocol for custom rule evaluation strategies."""

    def evaluate(self, context: dict[str, Any]) -> bool: ...


class BehaviorEngine:
    """Config-driven behavior interpreter.

    Features:
    - Pattern matching from YAML
    - Dynamic rule evaluation
    - Strategy selection
    - Intent classification
    - Tool routing
    - Task classification

    Usage:
        from src.brain.behavior_engine import behavior_engine

        # Classify intent
        result = behavior_engine.classify_intent("Привіт!", {})

        # Route tool
        server, tool, args = behavior_engine.route_tool("execute_command", {"cmd": "ls"})

        # Match pattern
        pattern = behavior_engine.match_pattern(
            context={'task_type': 'web', 'repeated_failures': True},
            pattern_type='adaptive_behavior'
        )
    """

    def __init__(self, config_path: Path | None = None):
        """Initialize behavior engine with config file.

        Args:
            config_path: Path to behavior_config.yaml (auto-detected if None)

        """
        if config_path is None:
            # Try multiple locations for dev and prod
            candidates = [
                Path(__file__).parent.parent.parent / "config" / "behavior_config.yaml",
                Path.home() / ".config" / "atlastrinity" / "behavior_config.yaml",
            ]
            for candidate in candidates:
                if candidate.exists():
                    config_path = candidate
                    break
            if config_path is None:
                config_path = candidates[0]  # Default to first option

        self.config_path = config_path
        self.config = self._load_config()
        self._pattern_cache: dict[str, list[Pattern]] = {}
        self._evaluators: dict[str, RuleEvaluator] = {}
        self._last_reload = time.time()

        # Performance metrics
        self._total_classifications = 0
        self._cache_hits = 0

        logger.info(f"[BEHAVIOR ENGINE] Initialized with config: {self.config_path}")

    def _load_config(self) -> dict[str, Any]:
        """Loads and validates behavior configuration."""
        try:
            if not self.config_path.exists():
                logger.warning(
                    f"[BEHAVIOR ENGINE] Config not found: {self.config_path}. Using empty config.",
                )
                return {}

            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                logger.info(
                    f"[BEHAVIOR ENGINE] Loaded config with {len(config.get('patterns', {}))} pattern groups",
                )
                return config
        except Exception as e:
            logger.error(f"[BEHAVIOR ENGINE] Failed to load config: {e}")
            return {}

    def reload_config(self) -> None:
        """Hot-reload configuration without restart."""
        logger.info("[BEHAVIOR ENGINE] Reloading configuration...")
        self.config = self._load_config()
        self._pattern_cache.clear()
        self._last_reload = time.time()
        logger.info("[BEHAVIOR ENGINE] Configuration reloaded successfully")

    def classify_intent(
        self, user_request: str, context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Classifies user intent based on config patterns.

        Args:
            user_request: User's input text
            context: Additional context for classification

        Returns:
            {
                'intent': 'chat' | 'task' | 'solo_task',
                'type': 'simple_chat' | 'info_query' | 'complex_task' | 'repeat_intent',
                'priority': 'high' | 'medium' | 'low',
                'use_deep_persona': bool,
                'require_tools': bool,
                'require_planning': bool
            }

        """
        self._total_classifications += 1
        context = context or {}
        request_lower = user_request.lower().strip()
        word_count = len(user_request.split())

        intent_config = self.config.get("intent_detection", {})

        # Priority 1: Repeat intent (highest priority)
        repeat_cfg = intent_config.get("repeat_intent", {})
        if any(kw in request_lower for kw in repeat_cfg.get("keywords", [])):
            return {
                "intent": repeat_cfg.get("intent", "task"),
                "type": "repeat_intent",
                "priority": repeat_cfg.get("priority", "high"),
                "resolve_from_memory": repeat_cfg.get("resolve_from_memory", True),
                "use_deep_persona": False,
                "require_tools": False,
                "require_planning": True,
            }

        # Priority 2: Philosophical query (Soul detection)
        philos_cfg = intent_config.get("philosophical_query", {})
        if any(kw in request_lower for kw in philos_cfg.get("keywords", [])):
            return {
                "intent": philos_cfg.get("intent", "chat"),
                "type": "philosophical_query",
                "priority": philos_cfg.get("priority", "highest"),
                "use_deep_persona": True,
                "require_tools": False,
                "require_planning": False,
            }

        # Priority 3: Simple chat (greetings)
        simple_cfg = intent_config.get("simple_chat", {})
        max_words = simple_cfg.get("max_words", 6)
        if word_count <= max_words and any(
            kw in request_lower for kw in simple_cfg.get("keywords", [])
        ):
            return {
                "intent": simple_cfg.get("intent", "chat"),
                "type": "simple_chat",
                "priority": simple_cfg.get("priority", "high"),
                "use_deep_persona": simple_cfg.get("use_deep_persona", False),
                "require_tools": False,
                "require_planning": False,
            }

        # Priority 3: Info queries (MUST come before complex_task)
        # Info queries like "погода у Львові" should trigger solo_task, not complex_task
        info_cfg = intent_config.get("info_query", {})
        if any(kw in request_lower for kw in info_cfg.get("keywords", [])):
            return {
                "intent": info_cfg.get("intent", "solo_task"),
                "type": "info_query",
                "priority": info_cfg.get("priority", "medium"),
                "use_deep_persona": info_cfg.get("use_deep_persona", False),
                "require_tools": info_cfg.get("require_tools", True),
                "require_planning": False,
            }

        # Priority 4: Complex tasks
        complex_cfg = intent_config.get("complex_task", {})
        indicators = complex_cfg.get("indicators", {})
        min_words = indicators.get("min_words", 7)
        action_verbs = indicators.get("contains_action_verbs", [])

        if word_count >= min_words or any(verb in request_lower for verb in action_verbs):
            return {
                "intent": complex_cfg.get("intent", "task"),
                "type": "complex_task",
                "priority": complex_cfg.get("priority", "high"),
                "use_deep_persona": complex_cfg.get("use_deep_persona", True),
                "require_tools": True,
                "require_planning": complex_cfg.get("require_planning", True),
                "enable_sequential_thinking": complex_cfg.get("enable_sequential_thinking", True),
            }

        # Default: Simple chat
        return {
            "intent": "chat",
            "type": "simple_chat",
            "priority": "medium",
            "use_deep_persona": False,
            "require_tools": False,
            "require_planning": False,
        }

    def select_strategy(self, task_type: str, context: dict[str, Any]) -> str:
        """Selects execution strategy from config.

        Args:
            task_type: Type of task (e.g., 'web_task', 'code_task')
            context: Contextual information for decision

        Returns:
            Strategy name (e.g., 'puppeteer-first', 'vibe-aggressive')

        """
        strategy_config = self.config.get("strategy_selection", {})
        task_strategies = strategy_config.get(task_type, {})

        # Evaluate context conditions
        for condition, strategy in task_strategies.items():
            if condition == "default":
                continue
            # Simple boolean context matching
            if context.get(condition.replace("_", "")) is True:
                logger.debug(
                    f"[BEHAVIOR ENGINE] Strategy selected: {strategy} (condition: {condition})",
                )
                return strategy

        # Return default or fallback
        default = task_strategies.get("default", "standard")
        logger.debug(f"[BEHAVIOR ENGINE] Strategy selected: {default} (default)")
        return default

    def route_tool(
        self, tool_name: str, args: dict[str, Any], explicit_server: str | None = None,
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Routes tool to appropriate server based on config.

        Args:
            tool_name: Name of the tool to route
            args: Tool arguments
            explicit_server: Explicitly requested server (optional)

        Returns:
            (server_name, resolved_tool_name, normalized_args)

        """
        tool_routing = self.config.get("tool_routing", {})
        tool_lower = tool_name.lower()

        # Check each routing category
        for category, config in tool_routing.items():
            synonyms = config.get("synonyms", [])

            # Check if tool matches this category
            if tool_lower in synonyms or any(tool_lower.startswith(syn) for syn in synonyms):
                priority_server = config.get("priority_server")
                # fallback_server = config.get("fallback_server")  # Reserved for future use

                # Check for special routing rules
                if "special_routing" in config:
                    for special_type, special_cfg in config["special_routing"].items():
                        keywords = special_cfg.get("keywords", [])
                        if any(kw in str(args).lower() for kw in keywords):
                            server = special_cfg.get("server")
                            tool = special_cfg.get("tool")
                            logger.info(
                                f"[BEHAVIOR ENGINE] Special routing: {tool_name} -> {server}.{tool}",
                            )
                            return server, tool, args

                # Check routing rules with patterns
                routing_rules = config.get("routing_rules", [])
                for rule in routing_rules:
                    pattern = rule.get("pattern", "")
                    if pattern and self._matches_pattern(tool_lower, pattern):
                        server = rule.get("server", priority_server)
                        resolved_tool = rule.get("tool", tool_name)
                        logger.debug(
                            f"[BEHAVIOR ENGINE] Rule match: {tool_name} -> {server}.{resolved_tool}",
                        )
                        return server, resolved_tool, args

                # Check tool mapping
                tool_mapping = config.get("tool_mapping", {})
                if tool_lower in tool_mapping:
                    resolved_tool = tool_mapping[tool_lower]
                    logger.debug(
                        f"[BEHAVIOR ENGINE] Mapping: {tool_name} -> {priority_server}.{resolved_tool}",
                    )
                    return priority_server, resolved_tool, args

                # Check action mapping (for macos-use)
                action_mapping = config.get("action_mapping", {})
                if tool_lower in action_mapping:
                    resolved_tool = action_mapping[tool_lower]
                    logger.debug(
                        f"[BEHAVIOR ENGINE] Action mapping: {tool_name} -> {priority_server}.{resolved_tool}",
                    )
                    return priority_server, resolved_tool, args

                # Use priority server
                if priority_server:
                    logger.debug(
                        f"[BEHAVIOR ENGINE] Priority server: {tool_name} -> {priority_server}.{tool_name}",
                    )
                    return priority_server, tool_name, args

        # No match found
        logger.warning(f"[BEHAVIOR ENGINE] No routing found for tool: {tool_name}")
        return explicit_server, tool_name, args

    def classify_task(self, task_description: str) -> list[str]:
        """Classifies task and returns recommended servers.

        Args:
            task_description: Description of the task

        Returns:
            List of recommended server names

        """
        task_lower = task_description.lower()
        task_classification = self.config.get("task_classification", {})

        # Match against all task types
        for task_type, config in task_classification.items():
            keywords = config.get("keywords", [])
            if any(kw in task_lower for kw in keywords):
                recommended = config.get("recommended_servers", [])
                logger.debug(f"[BEHAVIOR ENGINE] Task classified as {task_type}: {recommended}")
                return recommended

        # Default fallback
        default_servers = ["macos-use", "filesystem"]
        logger.debug(f"[BEHAVIOR ENGINE] Task classification fallback: {default_servers}")
        return default_servers

    def match_pattern(
        self, context: dict[str, Any], pattern_type: str, confidence_threshold: float = 0.6,
    ) -> Pattern | None:
        """Matches context against configured patterns.

        Args:
            context: Current execution context
            pattern_type: Type of pattern to match (e.g., 'adaptive_behavior')
            confidence_threshold: Minimum confidence to accept pattern

        Returns:
            Matched Pattern or None

        """
        # Check cache first
        cache_key = f"{pattern_type}_{hash(frozenset(context.items()))}"
        if cache_key in self._pattern_cache:
            self._cache_hits += 1

        patterns_config = self.config.get("patterns", {}).get(pattern_type, {})

        for pattern_name, pattern_cfg in patterns_config.items():
            trigger = pattern_cfg.get("trigger", {})
            conditions = trigger.get("conditions", {})
            min_confidence = trigger.get("min_confidence", 0.6)

            # Check if all conditions match
            if self._evaluate_conditions(conditions, context):
                metadata = pattern_cfg.get("metadata", {})
                initial_confidence = metadata.get("initial_confidence", min_confidence)

                if initial_confidence >= confidence_threshold:
                    pattern = Pattern(
                        name=pattern_name,
                        description=pattern_cfg.get("description", ""),
                        trigger=trigger,
                        action=pattern_cfg.get("action", {}),
                        confidence=initial_confidence,
                    )
                    logger.info(
                        f"[BEHAVIOR ENGINE] Pattern matched: {pattern_name} (confidence: {initial_confidence})",
                    )
                    return pattern

        return None

    def _evaluate_conditions(self, conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        """Evaluates if conditions match context."""
        for key, expected in conditions.items():
            if key == "error_contains":
                # Special case: substring matching in error
                error = str(context.get("error", "")).lower()
                if expected.lower() not in error:
                    return False
            else:
                # Exact match
                actual = context.get(key)
                if actual != expected:
                    return False
        return True

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Simple pattern matching (supports wildcards)."""
        import re

        # Convert glob-style pattern to regex
        regex_pattern = pattern.replace(".*", ".*").replace("*", ".*")
        return bool(re.match(regex_pattern, text))

    def evaluate_rule(self, rule_name: str, context: dict[str, Any]) -> Any:
        """Evaluates a rule from configuration.

        Args:
            rule_name: Name of rule in orchestration_flow.rules
            context: Context for evaluation

        Returns:
            The result of the first matching condition or default.

        """
        rules_config = self.config.get("orchestration_flow", {}).get("rules", {})
        rule = rules_config.get(rule_name, [])

        for entry in rule:
            if "condition" in entry:
                condition = entry["condition"]
                # Evaluate condition based on context
                if context.get(condition) or context.get(condition.replace("_", "")) is True:
                    return entry.get("result")
            elif "default" in entry:
                return entry.get("default")

        return None

    def get_output_processing(self, category: str) -> dict[str, Any]:
        """Returns output processing rules for a category."""
        return self.config.get("output_processing", {}).get(category, {})

    def get_background_monitoring(self, task_name: str) -> dict[str, Any]:
        """Returns background monitoring config for a task."""
        return self.config.get("background_monitoring", {}).get(task_name, {})

    def get_stats(self) -> dict[str, Any]:
        """Returns usage statistics."""
        cache_hit_rate = (
            (self._cache_hits / self._total_classifications * 100)
            if self._total_classifications > 0
            else 0
        )

        return {
            "total_classifications": self._total_classifications,
            "cache_hits": self._cache_hits,
            "cache_hit_rate_pct": round(cache_hit_rate, 2),
            "last_reload": self._last_reload,
            "config_path": str(self.config_path),
            "config_loaded": bool(self.config),
        }


class WorkflowEngine:
    """Deterministic Finite State Machine for executing workflows defined in config.
    """

    def __init__(self, behavior_engine_instance: BehaviorEngine):
        self.be = behavior_engine_instance

    async def execute_workflow(self, workflow_name: str, context: dict[str, Any]) -> bool:
        """Executes a workflow defined in behavior config.

        Args:
            workflow_name: Name of workflow (e.g. 'startup', 'error_recovery')
            context: Execution context (must contain 'orchestrator' for internal actions)

        Returns:
            Success status

        """
        workflow_config = self.be.config.get("workflows", {}).get(workflow_name)
        if not workflow_config:
            logger.warning(f"[WORKFLOW] Workflow '{workflow_name}' not found in config.")
            return False

        logger.info(f"[WORKFLOW] Starting workflow: {workflow_name}")
        stages = workflow_config.get("stages", [])

        try:
            for stage in stages:
                stage_name = stage.get("name", "unnamed")
                logger.info(f"[WORKFLOW] Entering stage: {stage_name}")

                steps = stage.get("steps", [])
                for step in steps:
                    await self._execute_step(step, context)

            logger.info(f"[WORKFLOW] Workflow '{workflow_name}' completed successfully.")
            return True

        except Exception as e:
            logger.error(f"[WORKFLOW] Workflow '{workflow_name}' failed: {e}")
            on_error = workflow_config.get("on_error", "continue")
            if on_error == "abort":
                raise e
            return False

    async def _execute_step(self, step: dict[str, Any], context: dict[str, Any]):
        """Executes a single workflow step."""
        # 1. Check Condition (simple boolean evaluation)
        if "if" in step:
            condition = step["if"]
            # Basic variable substitution for boolean check
            # Real implementation would need a safer eval or expression parser
            # For now, we support direct boolean values from context keys
            # e.g. "${error_analysis.can_auto_fix}" -> context['error_analysis']['can_auto_fix']

            # Simple variable resolution logic
            clean_cond = condition.strip("${} ")
            parts = clean_cond.split(".")
            val = context
            try:
                for part in parts:
                    val = val.get(part, {})

                result = bool(val)
            except Exception:
                result = False

            if not result:
                # Execute 'else' block if present
                if "else" in step:
                    await self._execute_step(step["else"], context)
                return

            # If condition met, execute 'then' block if present, or just the action logic below
            if "then" in step:
                await self._execute_step(step["then"], context)
                return

        # 2. Execute Action
        action_name = step.get("action")
        if not action_name:
            return

        params = step.get("params", {})
        # Resolve params (regex substitution)
        import re

        resolved_params = {}

        def _resolve_val(v):
            if isinstance(v, str):
                # Replace ${var.path} with context values
                def replacer(match):
                    path = match.group(1)
                    parts = path.split(".")
                    curr = context
                    try:
                        for p in parts:
                            curr = curr.get(p, {})
                        return str(curr)
                    except Exception:
                        return match.group(0)

                return re.sub(r"\$\{([^}]+)\}", replacer, v)
            elif isinstance(v, dict):
                return {ik: _resolve_val(iv) for ik, iv in v.items()}
            elif isinstance(v, list):
                return [_resolve_val(iv) for iv in v]
            return v

        for k, v in params.items():
            resolved_params[k] = _resolve_val(v)

        if action_name.startswith("internal."):
            # Execute Internal Action
            from .internal_actions import get_action

            func = get_action(action_name)
            if func:
                # Map 'async' param to 'async_warmup' to avoid keyword conflict if needed,
                # but our internal_actions use specific kwargs.
                # We pass context + kwargs.
                if asyncio.iscoroutinefunction(func):
                    await func(context, **resolved_params)
                else:
                    func(context, **resolved_params)
            else:
                logger.warning(f"[WORKFLOW] Internal action '{action_name}' not registered.")
        else:
            # Fallback: Treat as MCP Tool (via context orchestrator -> mcp_manager)
            # This requires the context to have access to tool execution capability
            # For startup workflows, we mostly use internal actions.
            pass


# Global singleton
behavior_engine = BehaviorEngine()
workflow_engine = WorkflowEngine(behavior_engine)
