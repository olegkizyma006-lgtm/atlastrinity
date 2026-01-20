import logging
from typing import Any

import pandas as pd

from src.brain.knowledge_graph import knowledge_graph

logger = logging.getLogger("brain.semantic_linker")


class SemanticLinker:
    """
    Analyzes datasets to find semantic intersections with existing data.
    Enables 'chaining' of tables via shared keys.
    """

    # Common semantic key patterns
    KEY_HINTS = [
        "id",
        "uid",
        "uuid",
        "guid",
        "email",
        "phone",
        "serial",
        "code",
        "sku",
        "isbn",
        "vin",
        "passport",
        "tax_id",
        "account",
        "product",
        "user",
        "client",
        "customer",
    ]

    async def discover_links(
        self, new_df: pd.DataFrame, new_node_id: str, namespace: str = "global"
    ) -> list[dict[str, Any]]:
        """
        Scans existing DATASET nodes to find potential overlaps.
        """
        links = []

        # 1. Identify potential keys in the new dataset
        new_keys = self._identify_possible_keys(new_df)
        if not new_keys:
            return []

        # 2. Fetch all other DATASET nodes in the same namespace (or global)
        existing_graph = await knowledge_graph.get_graph_data(namespace=namespace)
        dataset_nodes = [
            n for n in existing_graph["nodes"] if n["type"] == "DATASET" and n["id"] != new_node_id
        ]

        for ds_node in dataset_nodes:
            ds_attrs = ds_node.get("attributes", {})
            ds_columns = [str(c).lower() for c in ds_attrs.get("columns", [])]

            for n_key in new_keys:
                for e_col in ds_columns:
                    # Check for exact match, substring match, or alias match
                    is_match = False
                    if n_key == e_col:
                        is_match = True
                    elif n_key in e_col or e_col in n_key:
                        # Ensure it's not too short (avoid 'i' in 'id' false positives)
                        if len(n_key) > 2 and len(e_col) > 2:
                            is_match = True

                    if is_match:
                        links.append(
                            {
                                "source": new_node_id,
                                "target": ds_node["id"],
                                "relation": "LINKED_TO",
                                "attributes": {
                                    "shared_key": n_key,
                                    "matched_with": e_col,
                                    "description": f"Semantic link discovered between '{n_key}' and '{e_col}'",
                                },
                            }
                        )
                        break  # Only one link per dataset pair for simplicity

        return links

    def _identify_possible_keys(self, df: pd.DataFrame) -> list[str]:
        """
        Uses heuristics to find columns that look like identifiers or linkable keys.
        """
        possible_keys = []
        for col in df.columns:
            col_lower = str(col).lower()
            # Heuristic A: Name matches common key patterns
            if any(hint in col_lower for hint in self.KEY_HINTS):
                possible_keys.append(col_lower)
            # Heuristic B: Column has high cardinality (many unique values)
            elif df[col].nunique() / len(df) > 0.9 and len(df) > 10:
                possible_keys.append(col_lower)

        return list(set(possible_keys))


semantic_linker = SemanticLinker()
