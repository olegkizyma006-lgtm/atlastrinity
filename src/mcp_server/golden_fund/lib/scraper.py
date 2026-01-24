"""
Data Scraper Module for Golden Fund

Provides functionality to scrape data from open data portals and save it.
Ported from etl_module/src/scraping/data_scraper.py
"""

import csv
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import requests
from bs4 import BeautifulSoup

# Initialize logger
logger = logging.getLogger("golden_fund.scraper")


class ScrapeFormat(Enum):
    """Supported output formats for scraped data."""

    CSV = "csv"
    JSON = "json"
    XML = "xml"


class ScrapeResult:
    """Result container for scraping operations."""

    def __init__(self, success: bool, data: Any | None = None, error: str | None = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now()
        self.metadata = {}

    def __repr__(self) -> str:
        if self.success:
            return f"ScrapeResult(success=True, records={len(self.data) if isinstance(self.data, list) else 1})"
        return f"ScrapeResult(success=False, error={self.error})"


class DataScraper:
    """
    Data scraper for open data portals.
    """

    def __init__(self, user_agent: str = "AtlasTrinity-GoldenFund/1.0"):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/json,application/xml,*/*",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        logger.info("DataScraper initialized")

    def scrape_web_page(self, url: str, timeout: int = 30) -> ScrapeResult:
        try:
            logger.info(f"Scraping URL: {url}")
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            result = ScrapeResult(True, data=soup)
            result.metadata = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", "unknown"),
            }
            return result

        except requests.exceptions.RequestException as e:
            return ScrapeResult(False, error=f"Failed to scrape {url}: {e!s}")
        except Exception as e:
            return ScrapeResult(False, error=f"Unexpected error scraping {url}: {e!s}")

    def scrape_api_endpoint(
        self, url: str, params: dict | None = None, timeout: int = 30
    ) -> ScrapeResult:
        try:
            logger.info(f"Scraping API endpoint: {url}")
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError:
                data = response.text

            result = ScrapeResult(True, data=data)
            result.metadata = {"url": url, "status_code": response.status_code}
            return result

        except Exception as e:
            return ScrapeResult(False, error=f"API scraping failed: {e!s}")

    def save_data(
        self, data: Any, file_path: str | Path, format: ScrapeFormat = ScrapeFormat.JSON
    ) -> ScrapeResult:
        file_path = Path(file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format == ScrapeFormat.JSON:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == ScrapeFormat.CSV and isinstance(data, list) and len(data) > 0:
                keys = data[0].keys()
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(data)
            elif format == ScrapeFormat.XML:
                # Simplified XML save
                root = ET.Element("data")
                for i, record in enumerate(data if isinstance(data, list) else [data]):
                    item = ET.SubElement(root, "item", {"id": str(i)})
                    if isinstance(record, dict):
                        for k, v in record.items():
                            child = ET.SubElement(item, k)
                            child.text = str(v)
                tree = ET.ElementTree(root)
                tree.write(file_path, encoding="utf-8", xml_declaration=True)
            else:
                return ScrapeResult(False, error=f"Unsupported format or data structure: {format}")

            return ScrapeResult(True, data=str(file_path))

        except Exception as e:
            return ScrapeResult(False, error=f"Failed to save data: {e!s}")
