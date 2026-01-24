"""
Ingestion Tool for Golden Fund
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..lib.parser import DataParser
from ..lib.scraper import DataScraper, ScrapeFormat

logger = logging.getLogger("golden_fund.tools.ingest")

# Define storage path
DATA_DIR = Path(__file__).parents[4] / "data" / "golden_fund"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)


async def ingest_dataset(
    url: str, type: str = "web_page", process_pipeline: list[str] | None = None
) -> str:
    """
    Ingest a dataset from a URL.

    Args:
        url: URL to ingest
        type: 'web_page', 'api', or 'file'
        process_pipeline: List of steps (e.g. ['parse', 'vectorize'])
    """
    scraper = DataScraper()
    parser = DataParser()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    logger.info(f"Starting ingestion run {run_id} for {url} ({type})")

    # 1. Scrape / Fetch
    if type == "api":
        result = scraper.scrape_api_endpoint(url)
    else:
        # Default to web page scraping
        result = scraper.scrape_web_page(url)

    if not result.success:
        return f"Ingestion failed during scraping: {result.error}"

    # 2. Save Raw Data
    if result.data:
        # Determine format for saving
        if isinstance(result.data, dict | list):
            save_fmt = ScrapeFormat.JSON
            ext = ".json"
        else:
            # If soup or string, probably save as text/html?
            # For now let's handle JSON focus from ETL module
            save_fmt = ScrapeFormat.JSON
            ext = ".json"
            # TODO: Handle HTML saving properly if needed
            if hasattr(result.data, "get_text"):
                result.data = {"content": result.data.get_text()}

        raw_file = RAW_DIR / f"{run_id}_raw{ext}"
        save_res = scraper.save_data(result.data, raw_file, save_fmt)

        if not save_res.success:
            return f"Failed to save raw data: {save_res.error}"

        logger.info(f"Saved raw data to {save_res.data}")
    else:
        return "No data retrieved"

    summary = f"Ingestion {run_id} successful. Raw data: {raw_file.name}."

    # 3. Process Pipeline
    if process_pipeline and "parse" in process_pipeline:
        parse_res = parser.parse(raw_file)
        if parse_res.success:
            summary += f" Parsed {len(parse_res.data) if isinstance(parse_res.data, list) else 'content'} records."
        else:
            summary += f" Parsing failed: {parse_res.error}"

    return summary
