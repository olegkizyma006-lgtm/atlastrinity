"""
Data Transformation Layer for Golden Fund
Ported from etl_module/src/transformation/data_transformer.py
"""

import logging
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("golden_fund.transformer")


class TransformResult:
    def __init__(self, success: bool, data: Any | None = None, error: str | None = None):
        self.success = success
        self.data = data
        self.error = error


class UnifiedSchema(BaseModel):
    """
    Unified conceptual schema for Golden Fund entities.
    Flexible enough to hold diverse data but enforced key metadata.
    """

    name: str = Field(..., description="Entity name or title")
    type: str = Field(default="unknown", description="Entity type (e.g. person, company, dataset)")
    content: Any = Field(default=None, description="Main content or payload")
    source_format: str = Field(..., description="Original format")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataTransformer:
    def __init__(self):
        self.schema = UnifiedSchema
        logger.info("DataTransformer initialized")

    def transform(
        self, data: dict[str, Any] | list[dict[str, Any]], source_format: str = "unknown"
    ) -> TransformResult:
        try:
            if isinstance(data, list):
                transformed = []
                for item in data:
                    res = self._transform_item(item, source_format)
                    if res:
                        transformed.append(res)
                return TransformResult(True, data=transformed)
            else:
                res = self._transform_item(data, source_format)
                return TransformResult(
                    True if res else False, data=res, error="Validation failed" if not res else None
                )
        except Exception as e:
            return TransformResult(False, error=f"Transformation error: {e}")

    def _transform_item(self, item: dict[str, Any], source_format: str) -> dict[str, Any] | None:
        try:
            # Heuristic mapping flexibility
            name = item.get("name") or item.get("title") or item.get("id") or "Untitled"

            # Construct unified record
            record = {
                "name": str(name),
                "type": item.get("type", "entity"),
                "content": item,  # Store full original
                "source_format": source_format,
                "metadata": {k: v for k, v in item.items() if k not in ["name", "title"]},
            }

            # Validate
            validated = self.schema(**record)
            return validated.model_dump()
        except ValidationError as e:
            logger.warning(f"Validation failed for item: {e}")
            return None
