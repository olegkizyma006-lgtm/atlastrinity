"""
Main Data Parser Interface

Provides a unified interface for parsing different data formats.
"""

import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"


class ParseResult:
    """Result container for parsed data."""
    
    def __init__(self, success: bool, data: Optional[Any] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = {}
    
    def __repr__(self) -> str:
        if self.success:
            return f"ParseResult(success=True, data_type={type(self.data).__name__})"
        return f"ParseResult(success=False, error={self.error})"


class DataParser:
    """
    Unified data parser for multiple formats.
    
    This class provides a facade pattern to parse different data formats
    using format-specific parsers.
    """
    
    def __init__(self):
        self._parsers = {}
        self._register_parsers()
    
    def _register_parsers(self) -> None:
        """Register all available format parsers."""
        formats = ["csv", "json", "xml"]
        
        for fmt in formats:
            try:
                module = importlib.import_module(f"etl_module.src.parsing.formats.{fmt}_parser")
                # Handle different naming conventions
                if fmt == "csv":
                    parser_class = getattr(module, "CSVParser")
                elif fmt == "json":
                    parser_class = getattr(module, "JSONParser")
                elif fmt == "xml":
                    parser_class = getattr(module, "XMLParser")
                else:
                    parser_class = getattr(module, f"{fmt.capitalize()}Parser")
                self._parsers[fmt] = parser_class()
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load {fmt} parser: {e}")
    
    def parse(self, file_path: Union[str, Path], format_hint: Optional[DataFormat] = None) -> ParseResult:
        """
        Parse data from a file.
        
        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint (auto-detects if not provided)
        
        Returns:
            ParseResult containing parsed data or error information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ParseResult(False, error=f"File not found: {file_path}")
        
        # Auto-detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format(file_path)
            if format_hint is None:
                return ParseResult(False, error=f"Could not detect format for: {file_path}")
        
        # Get the appropriate parser
        parser = self._parsers.get(format_hint.value)
        if parser is None:
            return ParseResult(False, error=f"No parser available for format: {format_hint.value}")
        
        # Parse the data
        try:
            result = parser.parse(file_path)
            return result
        except Exception as e:
            return ParseResult(False, error=f"Parsing failed: {str(e)}")
    
    def _detect_format(self, file_path: Path) -> Optional[DataFormat]:
        """Detect file format based on file extension."""
        suffix = file_path.suffix.lower()
        
        format_map = {
            ".csv": DataFormat.CSV,
            ".json": DataFormat.JSON,
            ".xml": DataFormat.XML,
        }
        
        return format_map.get(suffix)
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return [fmt.value for fmt in DataFormat]
    
    def parse_to_dataframe(self, file_path: Union[str, Path], format_hint: Optional[DataFormat] = None) -> ParseResult:
        """
        Parse data and return as pandas DataFrame.
        
        Args:
            file_path: Path to the file to parse
            format_hint: Optional format hint
        
        Returns:
            ParseResult containing DataFrame or error
        """
        result = self.parse(file_path, format_hint)
        
        if not result.success:
            return result
        
        # Convert to DataFrame if not already
        if isinstance(result.data, pd.DataFrame):
            return result
        
        try:
            if isinstance(result.data, list):
                df = pd.DataFrame(result.data)
            elif isinstance(result.data, dict):
                df = pd.DataFrame([result.data])
            else:
                df = pd.DataFrame([result.data])
            
            result.data = df
            return result
        except Exception as e:
            return ParseResult(False, error=f"Failed to convert to DataFrame: {str(e)}")