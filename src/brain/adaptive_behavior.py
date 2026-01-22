"""
Adaptive Behavior Module

Provides non-standard algorithmic patterns for agent decision-making:
- Dynamic strategy selection
- Contextual deviation approval
- Learning-based behavior adjustment
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BehaviorPattern:
    """Represents a learned behavior pattern."""

    name: str
    trigger_conditions: dict[str, Any]
    action_override: dict[str, Any]
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AdaptiveBehaviorEngine:
    """
    Manages non-standard algorithmic behavior for agents.

    Features:
    - Pattern matching for deviation decisions
    - Context-aware strategy selection
    - Learning from past decisions
    """

    def __init__(self) -> None:
        # Legacy patterns dict kept for backward compatibility with record_deviation_outcome
        self.patterns: dict[str, BehaviorPattern] = {}
        self.deviation_history: list[dict[str, Any]] = []
        self.strategy_cache: dict[str, str] = {}

        # Behavior engine replaces hardcoded patterns
        from .behavior_engine import behavior_engine
        from .logger import logger

        self._engine = behavior_engine

        logger.info("[ADAPTIVE BEHAVIOR] Using config-driven behavior engine")

    # REMOVED: Patterns now defined in config/behavior_config.yaml
    # See behavior_config.yaml: patterns.adaptive_behavior section

    def should_deviate(
        self,
        current_step: dict[str, Any],
        context: dict[str, Any],
        confidence_threshold: float = 0.6,
    ) -> tuple[bool, str]:
        """
        Determines if the agent should deviate from standard behavior.
        Now delegates to BehaviorEngine for config-driven pattern matching.

        Returns:
            (should_deviate: bool, reason: str)
        """
        # Delegate to behavior engine for pattern matching
        combined_context = {**context, "step": current_step}
        pattern = self._engine.match_pattern(
            context=combined_context,
            pattern_type="adaptive_behavior",
            confidence_threshold=confidence_threshold,
        )

        if pattern:
            return True, f"Pattern '{pattern.name}' matched with confidence {pattern.confidence}"

        # Contextual deviation signals (kept for backward compatibility)
        if context.get("repeated_failures", 0) >= 3:
            return True, "Multiple failures detected, deviation recommended"

        if context.get("goal_alignment_score", 1.0) < 0.3:
            return True, "Low goal alignment, deviation recommended"

        return False, "Standard behavior recommended"

    def _matches_trigger(
        self,
        conditions: dict[str, Any],
        step: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Checks if conditions match current context."""
        for key, expected in conditions.items():
            if key == "error_contains":
                # Special case: check if error message contains substring
                error = str(context.get("error", "") or step.get("error", "")).lower()
                if expected.lower() not in error:
                    return False
            else:
                actual = step.get(key) or context.get(key)
                if actual != expected:
                    return False
        return True

    def record_deviation_outcome(
        self,
        pattern_name: str,
        success: bool,
        context: dict[str, Any],
    ) -> None:
        """Records the outcome of a deviation for learning."""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            pattern.usage_count += 1
            # Update success rate with exponential moving average
            alpha = 0.3
            new_success = 1.0 if success else 0.0
            pattern.success_rate = alpha * new_success + (1 - alpha) * pattern.success_rate
            # Adjust confidence based on performance
            if pattern.success_rate > 0.8 and pattern.usage_count > 5:
                pattern.confidence = min(pattern.confidence + 0.05, 1.0)
            elif pattern.success_rate < 0.4 and pattern.usage_count > 5:
                pattern.confidence = max(pattern.confidence - 0.1, 0.0)

        self.deviation_history.append(
            {
                "pattern": pattern_name,
                "success": success,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def register_pattern(
        self,
        name: str,
        trigger_conditions: dict[str, Any],
        action_override: dict[str, Any],
        initial_confidence: float = 0.5,
    ) -> None:
        """Registers a new behavior pattern."""
        self.patterns[name] = BehaviorPattern(
            name=name,
            trigger_conditions=trigger_conditions,
            action_override=action_override,
            confidence=initial_confidence,
        )

    def get_strategy_recommendation(
        self,
        task_type: str,
        context: dict[str, Any],
    ) -> str:
        """
        Returns a strategy recommendation based on task type and context.
        Now delegates to BehaviorEngine for config-driven selection.
        """
        # Create cache key from task type and relevant context
        context_key_parts = sorted(
            [f"{k}={v}" for k, v in context.items() if isinstance(v, str | bool | int)]
        )
        cache_key = f"{task_type}:{':'.join(context_key_parts[:5])}"

        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]

        # Delegate to behavior engine
        strategy = self._engine.select_strategy(task_type, context)
        self.strategy_cache[cache_key] = strategy
        return strategy

    def get_pattern_stats(self) -> dict[str, Any]:
        """Returns statistics about registered patterns."""
        return {
            "total_patterns": len(self.patterns),
            "deviation_history_count": len(self.deviation_history),
            "patterns": {
                name: {
                    "confidence": p.confidence,
                    "usage_count": p.usage_count,
                    "success_rate": p.success_rate,
                }
                for name, p in self.patterns.items()
            },
        }


# Global singleton
adaptive_behavior = AdaptiveBehaviorEngine()
