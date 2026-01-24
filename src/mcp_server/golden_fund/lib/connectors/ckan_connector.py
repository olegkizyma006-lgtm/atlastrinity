"""
CKAN Connector for Golden Fund
Handles interaction with CKAN-based open data portals (e.g., data.gov.ua).
"""

import logging
from datetime import datetime
from typing import Any, Optional

import requests

logger = logging.getLogger("golden_fund.connectors.ckan")


class CKANConnector:
    def __init__(self, base_url: str = "https://data.gov.ua/api/3"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "AtlasTrinity-GoldenFund/1.0", "Accept": "application/json"}
        )
        logger.info(f"CKAN Connector initialized for {self.base_url}")

    def search_packages(self, query: str, rows: int = 10) -> list[dict[str, Any]]:
        """
        Search for datasets (packages) in CKAN.
        """
        url = f"{self.base_url}/action/package_search"
        params = {"q": query, "rows": rows}

        try:
            logger.info(f"Searching CKAN packages for: {query}")
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                results = data["result"]["results"]
                logger.info(f"Found {len(results)} packages")
                return results
            else:
                logger.warning(f"CKAN search failed: {data.get('error')}")
                return []

        except Exception as e:
            logger.error(f"Error searching CKAN: {e}")
            return []

    def get_package_details(self, package_id: str) -> dict[str, Any] | None:
        """
        Get details for a specific package.
        """
        url = f"{self.base_url}/action/package_show"
        params = {"id": package_id}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                return data["result"]
            return None

        except Exception as e:
            logger.error(f"Error fetching package {package_id}: {e}")
            return None

    def get_resource_url(self, resource: dict[str, Any]) -> str:
        """
        Extract the download URL for a resource.
        """
        return str(resource.get("url") or "")

    def find_resources_by_format(
        self, package: dict[str, Any], formats: list[str] | None = None
    ) -> list[dict[str, Any]]:
        if formats is None:
            formats = ["CSV", "JSON", "XML"]
        """
        Find resources in a package matching specific formats.
        """
        resources = package.get("resources", [])
        matched = [
            res
            for res in resources
            if res.get("format", "").upper() in [f.upper() for f in formats]
        ]
        return matched
