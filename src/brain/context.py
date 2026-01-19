"""
AtlasTrinity Shared Context

Singleton module for sharing context between all agents.
Solves the problem of agents using wrong paths or lacking awareness
of the current working directory and recent operations.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

# Get the actual user home directory
ACTUAL_HOME = os.path.expanduser("~")
GITHUB_ROOT = f"{ACTUAL_HOME}/Documents/GitHub"


@dataclass
class SharedContext:
    """
    Shared context singleton that all agents can access.

    Provides:
    - Current working directory awareness
    - Recent file tracking
    - Last successful path memory
    - Project context
    - Goal tracking for agent coordination
    """

    # Core path context - uses actual user home directory
    current_working_directory: str = GITHUB_ROOT
    active_project: str = ""
    last_successful_path: str = ""

    # System environment context
    home_directory: str = ACTUAL_HOME
    applications_directory: str = "/Applications"
    documents_directory: str = f"{ACTUAL_HOME}/Documents"
    downloads_directory: str = f"{ACTUAL_HOME}/Downloads"
    desktop_directory: str = f"{ACTUAL_HOME}/Desktop"
    
    # Execution context
    is_packaged: bool = False

    # File tracking
    recent_files: List[str] = field(default_factory=list)
    created_directories: List[str] = field(default_factory=list)

    # Operation history (for debugging)
    operation_count: int = 0
    last_operation: str = ""
    last_update: Optional[datetime] = None
    available_tools_summary: str = ""

    # Goal tracking for agent coordination
    current_goal: str = ""
    parent_goal: Optional[str] = None
    goal_stack: List[str] = field(default_factory=list)
    recursive_depth: int = 0
    max_recursive_depth: int = 5
    current_step_id: Optional[int] = None
    total_steps: int = 0
    available_mcp_catalog: str = ""

    def __post_init__(self):
        # Detect if application is packaged (binary/app mode)
        import sys
        self.is_packaged = getattr(sys, "frozen", False)
        
        # Auto-detect CWD on startup
        try:
            cwd = os.getcwd()
            if "/Documents/GitHub/" in cwd:
                self.current_working_directory = cwd
                self.last_successful_path = cwd
                # Infer project
                parts = cwd.split("/Documents/GitHub/")
                if len(parts) > 1:
                    self.active_project = parts[1].split("/")[0]
        except Exception:
            pass

    def update_path(self, path: str, operation: str = "access") -> None:
        """
        Update context with a new successful path.
        Called by agents after successful operations.
        """
        if path and (path.startswith("/Users") or path.startswith("~")):
            path = os.path.expandvars(os.path.expanduser(path))
            self.last_successful_path = path

            # Infer project from path
            if "/Documents/GitHub/" in path:
                parts = path.split("/Documents/GitHub/")
                if len(parts) > 1:
                    project_part = parts[1].split("/")[0]
                    if project_part:
                        self.active_project = project_part
                        self.current_working_directory = f"{GITHUB_ROOT}/{project_part}"

            # Track files
            if operation in ["create", "write", "read"]:
                if path not in self.recent_files:
                    self.recent_files.append(path)
                    # Keep only last 20 files
                    if len(self.recent_files) > 20:
                        self.recent_files = self.recent_files[-20:]

            # Track directories
            if operation == "create_directory":
                if path not in self.created_directories:
                    self.created_directories.append(path)

            self.operation_count += 1
            self.last_operation = f"{operation}: {path}"
            self.last_update = datetime.now()

    def get_best_path(self, hint: str = "") -> str:
        """
        Get the most likely correct path based on context.
        Used by agents to auto-correct placeholder paths.
        """
        # If we have an active project, use that
        if self.active_project:
            return self.current_working_directory

        # If we have a last successful path, use its directory
        if self.last_successful_path:
            import os

            return os.path.dirname(self.last_successful_path)

        # Default to GitHub directory
        return GITHUB_ROOT

    def resolve_path(self, raw_path: str) -> str:
        """
        Resolve a potentially invalid path to a valid one.
        Handles placeholders, tilde, relative paths.
        """
        if not raw_path:
            return self.get_best_path()

        raw_path = os.path.expandvars(raw_path)

        # Expand tilde
        if raw_path.startswith("~/"):
            raw_path = raw_path.replace("~/", f"{ACTUAL_HOME}/")
        elif raw_path == "~":
            raw_path = ACTUAL_HOME

        # Check for placeholder patterns
        placeholder_patterns = ["/path/", "/to/", "${", "{{"]
        if any(p in raw_path for p in placeholder_patterns):
            return self.get_best_path()

        # Check for valid absolute path
        if not raw_path.startswith("/Users") and not raw_path.startswith("/Applications"):
            # Relative path - prepend working directory
            return f"{self.current_working_directory}/{raw_path.lstrip('/')}"

        return raw_path

    def to_dict(self) -> dict:
        """Export context for logging or debugging."""
        return {
            "cwd": self.current_working_directory,
            "project": self.active_project,
            "last_path": self.last_successful_path,
            "recent_files": self.recent_files[-5:],
            "system_paths": {
                "home": self.home_directory,
                "applications": self.applications_directory,
                "documents": self.documents_directory,
                "downloads": self.downloads_directory,
                "desktop": self.desktop_directory
            },
            "is_packaged": self.is_packaged,
            "operation_count": self.operation_count,
            "last_op": self.last_operation,
            # Goal tracking
            "current_goal": self.current_goal,
            "parent_goal": self.parent_goal,
            "goal_depth": len(self.goal_stack),
            "recursive_depth": self.recursive_depth,
            "step_progress": f"{self.current_step_id}/{self.total_steps}" if self.total_steps else "—",
        }

    def push_goal(self, goal: str, total_steps: int = 0) -> None:
        """
        Push a new goal onto the stack (entering a sub-task).
        Called by Atlas when creating a new plan.
        """
        if self.current_goal:
            self.goal_stack.append(self.current_goal)
            self.parent_goal = self.current_goal
        self.current_goal = goal
        self.total_steps = total_steps
        self.current_step_id = 0
        self.recursive_depth = len(self.goal_stack)

    def pop_goal(self) -> str:
        """
        Pop the current goal from the stack (leaving a sub-task).
        Returns the goal that was popped.
        """
        completed_goal = self.current_goal
        if self.goal_stack:
            self.current_goal = self.goal_stack.pop()
            self.parent_goal = self.goal_stack[-1] if self.goal_stack else None
        else:
            self.current_goal = ""
            self.parent_goal = None
        self.recursive_depth = len(self.goal_stack)
        self.total_steps = 0
        self.current_step_id = None
        return completed_goal

    def advance_step(self) -> None:
        """Advance the current step counter."""
        if self.current_step_id is not None:
            self.current_step_id += 1

    def get_goal_context(self) -> str:
        """
        Get formatted goal context string for agent prompts.
        This helps agents understand the current task hierarchy.
        """
        if not self.current_goal:
            return ""

        lines = []
        
        # Build Hierarchy Path
        hierarchy = []
        if self.goal_stack:
            hierarchy = self.goal_stack + [self.current_goal]
        else:
            hierarchy = [self.current_goal]
            
        lines.append("🎯 GOAL HIERARCHY:")
        for idx, g in enumerate(hierarchy):
            indent = "  " * idx
            prefix = "└─" if idx > 0 else "●"
            lines.append(f"{indent}{prefix} {g}")
        
        if self.total_steps > 0:
            lines.append(f"\n📝 PROGRESS: Step {self.current_step_id or 0}/{self.total_steps}")
            
        if self.recursive_depth > 0:
            lines.append(f"🔄 RECURSION DEPTH: {self.recursive_depth} (max: {self.max_recursive_depth})")

        return "\n".join(lines)

    def is_at_max_depth(self) -> bool:
        """Check if we've reached maximum recursion depth."""
        return self.recursive_depth >= self.max_recursive_depth


# Singleton instance - import this in other modules
shared_context = SharedContext()
