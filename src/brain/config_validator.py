"""
AtlasTrinity Configuration Validator

Schema-based validation for:
- config.yaml structure and values
- config.json MCP server definitions
- Environment variable placeholders
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import CONFIG_ROOT, MCP_DIR, PROJECT_ROOT
from .logger import logger


@dataclass
class ValidationIssue:
    """A single validation issue."""

    level: str  # "error", "warning", "info"
    path: str  # Config path like "agents.atlas.model"
    message: str
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validating a configuration file."""

    file_path: Path
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]


# Expected structure schemas
YAML_SCHEMA = {
    "agents": {
        "_type": "dict",
        "_required": True,
        "atlas": {
            "_type": "dict",
            "model": {"_type": "str"},
            "temperature": {"_type": "float", "_range": (0.0, 2.0)},
            "max_tokens": {"_type": "int", "_range": (100, 100000)},
        },
        "tetyana": {
            "_type": "dict",
            "model": {"_type": "str"},
            "temperature": {"_type": "float", "_range": (0.0, 2.0)},
            "max_tokens": {"_type": "int", "_range": (100, 100000)},
        },
        "grisha": {
            "_type": "dict",
            "vision_model": {"_type": "str"},
            "temperature": {"_type": "float", "_range": (0.0, 2.0)},
            "max_tokens": {"_type": "int", "_range": (100, 100000)},
        },
    },
    "orchestrator": {
        "_type": "dict",
        "max_recursion_depth": {"_type": "int", "_range": (1, 20)},
        "task_timeout": {"_type": "int", "_range": (10, 3600)},
        "subtask_timeout": {"_type": "int", "_range": (10, 1800)},
        "recovery_voice_agent": {"_type": "str", "_enum": ["atlas", "grisha"]},
        "validate_failed_steps_with_grisha": {"_type": "bool"},
    },
    "mcp": {"_type": "dict"},
    "security": {
        "_type": "dict",
        "dangerous_commands": {"_type": "list"},
        "require_confirmation": {"_type": "bool"},
    },
    "voice": {"_type": "dict"},
    "logging": {
        "_type": "dict",
        "level": {"_type": "str", "_enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
    },
}

MCP_SERVER_SCHEMA = {
    "command": {"_type": "str", "_required": True},
    "args": {"_type": "list"},
    "env": {"_type": "dict"},
    "description": {"_type": "str"},
    "disabled": {"_type": "bool"},
    "tier": {"_type": "int", "_range": (1, 4)},
    "agents": {"_type": "list"},
    "connect_timeout": {"_type": "int", "_range": (1, 600)},
}


class ConfigValidator:
    """Validates AtlasTrinity configuration files."""

    def __init__(self):
        self.env_var_pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def _check_range(self, value: Any, range_tuple: tuple) -> bool:
        """Check if numeric value is within range."""
        if not isinstance(value, (int, float)):
            return True  # Skip non-numeric
        min_val, max_val = range_tuple
        return min_val <= value <= max_val

    def _check_enum(self, value: Any, allowed: list) -> bool:
        """Check if value is in allowed list."""
        return value in allowed

    def _validate_against_schema(
        self,
        data: dict,
        schema: dict,
        path: str = "",
        issues: list[ValidationIssue] | None = None,
    ) -> list[ValidationIssue]:
        """Recursively validate data against schema."""
        if issues is None:
            issues = []

        for key, spec in schema.items():
            if key.startswith("_"):
                continue

            current_path = f"{path}.{key}" if path else key

            if key not in data:
                if spec.get("_required"):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            path=current_path,
                            message="Required key missing",
                        )
                    )
                continue

            value = data[key]

            # Type check
            expected_type = spec.get("_type")
            if expected_type and not self._check_type(value, expected_type):
                issues.append(
                    ValidationIssue(
                        level="error",
                        path=current_path,
                        message=f"Expected type '{expected_type}', got '{type(value).__name__}'",
                        value=value,
                    )
                )
                continue

            # Range check
            if "_range" in spec and not self._check_range(value, spec["_range"]):
                issues.append(
                    ValidationIssue(
                        level="warning",
                        path=current_path,
                        message=f"Value {value} outside range {spec['_range']}",
                        value=value,
                    )
                )

            # Enum check
            if "_enum" in spec and not self._check_enum(value, spec["_enum"]):
                issues.append(
                    ValidationIssue(
                        level="warning",
                        path=current_path,
                        message=f"Value '{value}' not in allowed: {spec['_enum']}",
                        value=value,
                    )
                )

            # Recurse into nested dicts
            if expected_type == "dict" and isinstance(value, dict):
                nested_spec = {k: v for k, v in spec.items() if not k.startswith("_")}
                if nested_spec:
                    self._validate_against_schema(value, nested_spec, current_path, issues)

        return issues

    def _check_env_vars(self, data: Any, path: str = "") -> list[ValidationIssue]:
        """Check for undefined environment variable placeholders."""
        issues = []

        if isinstance(data, str):
            matches = self.env_var_pattern.findall(data)
            for var_name in matches:
                if not os.environ.get(var_name):
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            path=path,
                            message=f"Environment variable ${{{var_name}}} not set",
                            value=data,
                        )
                    )
        elif isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                issues.extend(self._check_env_vars(value, current_path))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                issues.extend(self._check_env_vars(item, current_path))

        return issues

    def validate_yaml(self, path: Path) -> ValidationResult:
        """Validate config.yaml file."""
        issues = []

        if not path.exists():
            return ValidationResult(
                file_path=path,
                valid=False,
                issues=[ValidationIssue(level="error", path=str(path), message="File not found")],
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            return ValidationResult(
                file_path=path,
                valid=False,
                issues=[
                    ValidationIssue(level="error", path=str(path), message=f"YAML parse error: {e}")
                ],
            )

        # Schema validation
        issues.extend(self._validate_against_schema(data, YAML_SCHEMA))

        # Environment variable check
        issues.extend(self._check_env_vars(data))

        return ValidationResult(
            file_path=path,
            valid=len([i for i in issues if i.level == "error"]) == 0,
            issues=issues,
        )

    def validate_mcp_json(self, path: Path) -> ValidationResult:
        """Validate MCP config.json file."""
        issues = []

        if not path.exists():
            return ValidationResult(
                file_path=path,
                valid=False,
                issues=[ValidationIssue(level="error", path=str(path), message="File not found")],
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return ValidationResult(
                file_path=path,
                valid=False,
                issues=[
                    ValidationIssue(level="error", path=str(path), message=f"JSON parse error: {e}")
                ],
            )

        servers = data.get("mcpServers", {})

        for server_name, server_config in servers.items():
            if server_name.startswith("_"):
                continue  # Skip comment keys

            server_path = f"mcpServers.{server_name}"

            # Validate against MCP server schema
            for key, spec in MCP_SERVER_SCHEMA.items():
                current_path = f"{server_path}.{key}"

                if key not in server_config:
                    if spec.get("_required"):
                        issues.append(
                            ValidationIssue(
                                level="error",
                                path=current_path,
                                message=f"Required key '{key}' missing",
                            )
                        )
                    continue

                value = server_config[key]
                expected_type = spec.get("_type")

                if (
                    expected_type
                    and isinstance(expected_type, str)
                    and not self._check_type(value, expected_type)
                ):
                    issues.append(
                        ValidationIssue(
                            level="error",
                            path=current_path,
                            message=f"Expected type '{expected_type}', got '{type(value).__name__}'",
                            value=value,
                        )
                    )

                if (
                    "_range" in spec
                    and isinstance(spec["_range"], tuple)
                    and not self._check_range(value, spec["_range"])
                ):
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            path=current_path,
                            message=f"Value {value} outside range {spec['_range']}",
                            value=value,
                        )
                    )

            # Environment variable check
            issues.extend(self._check_env_vars(server_config, server_path))

        return ValidationResult(
            file_path=path,
            valid=len([i for i in issues if i.level == "error"]) == 0,
            issues=issues,
        )

    def validate_all(self) -> list[ValidationResult]:
        """Validate all configuration files."""
        results = []

        # Global config.yaml
        yaml_path = CONFIG_ROOT / "config.yaml"
        if yaml_path.exists():
            results.append(self.validate_yaml(yaml_path))

        # Project config.yaml
        project_yaml = PROJECT_ROOT / "config" / "config.yaml"
        if project_yaml.exists():
            results.append(self.validate_yaml(project_yaml))

        # Global MCP config
        mcp_json = MCP_DIR / "config.json"
        if mcp_json.exists():
            results.append(self.validate_mcp_json(mcp_json))

        # Project MCP config
        project_mcp = PROJECT_ROOT / "src" / "mcp_server" / "config.json"
        if project_mcp.exists():
            results.append(self.validate_mcp_json(project_mcp))

        return results

    def log_results(self, results: list[ValidationResult]) -> bool:
        """Log validation results and return True if all valid."""
        all_valid = True

        for result in results:
            if result.valid:
                logger.info(f"[ConfigValidator] ✓ {result.file_path.name} is valid")
            else:
                all_valid = False
                logger.error(f"[ConfigValidator] ✗ {result.file_path.name} has errors")

            for issue in result.issues:
                if issue.level == "error":
                    logger.error(f"  [{issue.path}] {issue.message}")
                elif issue.level == "warning":
                    logger.warning(f"  [{issue.path}] {issue.message}")
                else:
                    logger.info(f"  [{issue.path}] {issue.message}")

        return all_valid


# Global instance
config_validator = ConfigValidator()
