"""
BaseAgent - Shared utilities for Trinity agents

This module provides common functionality used by Atlas, Tetyana, and Grisha agents.
"""

import json
from typing import Any, Dict


class BaseAgent:
    """Base class for Trinity agents with shared utilities."""
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
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
