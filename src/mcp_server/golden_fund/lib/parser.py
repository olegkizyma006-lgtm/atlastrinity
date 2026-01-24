"""
Main Data Parser Interface for Golden Fund
Ported from etl_module/src/parsing/data_parser.py
"""

from pathlib import Path
from typing import Optional, Union

from .formats import CSVParser, JSONParser, ParseResult, XMLParser


class DataParser:
    def __init__(self):
        self._parsers = {"json": JSONParser(), "csv": CSVParser(), "xml": XMLParser()}

    def parse(self, file_path: str | Path, format_hint: str | None = None) -> ParseResult:
        file_path = Path(file_path)

        if not file_path.exists():
            return ParseResult(False, error=f"File not found: {file_path}")

        if format_hint is None:
            format_hint = file_path.suffix.lstrip(".").lower()

        parser = self._parsers.get(format_hint)
        if not parser:
            return ParseResult(False, error=f"No parser for format: {format_hint}")

        return parser.parse(file_path)
