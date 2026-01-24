import logging
from typing import Any

import pandas as pd

logger = logging.getLogger("brain.data_guard")


class DataQualityGuard:
    """Validates external datasets before ingestion into the Golden Fund.
    Provides checks for completeness, structural integrity, and data sanity.
    """

    def __init__(self, completeness_threshold: float = 0.7, max_null_ratio: float = 0.3):
        self.completeness_threshold = completeness_threshold
        self.max_null_ratio = max_null_ratio

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
        """Performs a multi-point validation of a pandas DataFrame."""
        report = {
            "dataset": dataset_name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "checks": {},
            "is_worthy": True,
            "issues": [],
        }

        if df.empty:
            report["is_worthy"] = False
            report["issues"].append("Dataset is empty")
            return report

        # 1. Completeness Check
        df.isnull().mean()
        overall_null_ratio = df.isnull().values.mean()
        report["checks"]["null_ratio"] = float(overall_null_ratio)

        if overall_null_ratio > self.max_null_ratio:
            report["is_worthy"] = False
            report["issues"].append(f"Excessive null values detected ({overall_null_ratio:.2%})")

        # 2. Structural Integrity (Check for mixed types in columns)
        for col in df.columns:
            types = df[col].map(type).unique()
            if len(types) > 1:
                # Filter out None/NaN types from the mix
                effective_types = [t for t in types if t is not type(None) and t is not float]
                if len(effective_types) > 1:
                    report["issues"].append(
                        f"Column '{col}' has mixed data types: {effective_types}",
                    )
                    # Mixed types are a warning, not necessarily a rejection, but we mark it
                    report["is_worthy"] = False if len(effective_types) > 2 else report["is_worthy"]

        # 3. "Nonsense" Detection (Min length / Entropy heuristics)
        avg_lengths = {}
        for col in df.select_dtypes(include=["object"]).columns:
            non_null = df[col].dropna()
            if not non_null.empty:
                avg_len = non_null.astype(str).str.len().mean()
                avg_lengths[col] = float(avg_len)
                if avg_len < 1:  # Obvious junk
                    report["is_worthy"] = False
                    report["issues"].append(f"Column '{col}' contains ultra-short or empty strings")

        report["checks"]["avg_string_lengths"] = avg_lengths

        # 4. Trash Detection (Special characters / Corrupted artifacts)
        # Using anchored patterns to avoid false positives in real text
        trash_patterns = [r"^\s*$", r"^undefined$", r"^NaN$", r"^NULL$", r"^\?+$"]
        trash_count = 0
        for col in df.select_dtypes(include=["object"]).columns:
            for pattern in trash_patterns:
                trash_count += df[col].astype(str).str.contains(pattern, regex=True, na=False).sum()

        report["checks"]["trash_count"] = int(trash_count)
        if trash_count > len(df) * 0.1:  # If more than 10% of cells look like trash
            report["is_worthy"] = False
            report["issues"].append(
                f"High volume of trash or corrupted markers detected ({trash_count} hits)",
            )

        return report


data_guard = DataQualityGuard()
