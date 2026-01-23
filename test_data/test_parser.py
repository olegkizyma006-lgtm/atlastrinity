#!/usr/bin/env python3

"""
Test script for the data parsing component.
"""

import sys
import os

# Add the etl_module to Python path
sys.path.insert(0, "/Users/dev/Documents/GitHub/atlastrinity")

from etl_module.src.parsing import DataParser, DataFormat

def test_csv_parsing():
    """Test CSV parsing functionality."""
    print("Testing CSV parsing...")
    parser = DataParser()
    result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.csv")
    
    if result.success:
        print(f"✓ CSV parsing successful: {type(result.data).__name__}")
        print(f"  Rows: {result.metadata.get('rows', 'N/A')}")
        print(f"  Columns: {result.metadata.get('columns', 'N/A')}")
        print(f"  Preview: {result.data.head(2).to_dict('records')}")
    else:
        print(f"✗ CSV parsing failed: {result.error}")
    print()

def test_json_parsing():
    """Test JSON parsing functionality."""
    print("Testing JSON parsing...")
    parser = DataParser()
    result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.json")
    
    if result.success:
        print(f"✓ JSON parsing successful: {type(result.data).__name__}")
        if hasattr(result.data, 'shape'):
            print(f"  Shape: {result.data.shape}")
        else:
            print(f"  Items: {result.metadata.get('items', 'N/A')}")
        print(f"  Preview: {str(result.data)[:100]}...")
    else:
        print(f"✗ JSON parsing failed: {result.error}")
    print()

def test_xml_parsing():
    """Test XML parsing functionality."""
    print("Testing XML parsing...")
    parser = DataParser()
    result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.xml")
    
    if result.success:
        print(f"✓ XML parsing successful: {type(result.data).__name__}")
        print(f"  Root tag: {result.metadata.get('root_tag', 'N/A')}")
        print(f"  Data keys: {list(result.data.keys())}")
    else:
        print(f"✗ XML parsing failed: {result.error}")
    print()

def test_dataframe_conversion():
    """Test DataFrame conversion functionality."""
    print("Testing DataFrame conversion...")
    parser = DataParser()
    
    # Test CSV to DataFrame
    result = parser.parse_to_dataframe("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.csv")
    print(f"CSV to DataFrame: {'✓' if result.success else '✗'} {result.error if not result.success else ''}")
    
    # Test JSON to DataFrame
    result = parser.parse_to_dataframe("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.json")
    print(f"JSON to DataFrame: {'✓' if result.success else '✗'} {result.error if not result.success else ''}")
    
    # Test XML to DataFrame
    result = parser.parse_to_dataframe("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.xml")
    print(f"XML to DataFrame: {'✓' if result.success else '✗'} {result.error if not result.success else ''}")
    print()

def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    parser = DataParser()
    
    # Test non-existent file
    result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/nonexistent.csv")
    print(f"Non-existent file: {'✓' if not result.success else '✗'} {result.error}")
    
    # Test unsupported format
    result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.csv", format_hint=DataFormat.XML)
    print(f"Wrong format hint: {'✓' if not result.success else '✗'} {result.error}")
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA PARSING COMPONENT TEST")
    print("=" * 60)
    print()
    
    try:
        test_csv_parsing()
        test_json_parsing()
        test_xml_parsing()
        test_dataframe_conversion()
        test_error_handling()
        
        print("=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()