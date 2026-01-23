"""
Scraping Module

Provides data scraping functionality for the ETL pipeline.
"""

from .data_scraper import DataScraper, ScrapeResult, ScrapeFormat, create_data_scraper

__all__ = ["DataScraper", "ScrapeResult", "ScrapeFormat", "create_data_scraper"]