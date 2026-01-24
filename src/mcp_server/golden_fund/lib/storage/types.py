from datetime import datetime
from typing import Any, Optional


class StorageResult:
    """Result container for storage operations."""

    def __init__(
        self, success: bool, target: str, data: Any | None = None, error: str | None = None
    ):
        self.success = success
        self.target = target
        self.data = data
        self.error = error
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        if self.success:
            return f"StorageResult(success=True, target={self.target})"
        return f"StorageResult(success=False, target={self.target}, error={self.error})"
