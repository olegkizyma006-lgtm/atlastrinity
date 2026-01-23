#!/usr/bin/env python3

"""
Test script for the data transformation component.
"""

import sys
import os

# Add the etl_module to Python path
sys.path.insert(0, "/Users/dev/Documents/GitHub/atlastrinity")

from etl_module.src.parsing import DataParser, DataFormat
from etl_module.src.transformation import DataTransformer, TransformResult

def test_csv_transformation():
    """Test CSV data transformation."""
    print("Testing CSV transformation...")
    
    # Parse CSV data
    parser = DataParser()
    parse_result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.csv")
    
    if not parse_result.success:
        print(f"✗ CSV parsing failed: {parse_result.error}")
        return
    
    # Transform parsed data
    transformer = DataTransformer()
    transform_result = transformer.transform_from_dataframe(parse_result.data, "csv")
    
    if transform_result.success:
        print(f"✓ CSV transformation successful: {type(transform_result.data).__name__}")
        print(f"  Records transformed: {len(transform_result.data)}")
        print(f"  Sample record: {transform_result.data[0]}")
        print(f"  Schema fields: {list(transform_result.data[0].keys())}")
    else:
        print(f"✗ CSV transformation failed: {transform_result.error}")
    print()

def test_json_transformation():
    """Test JSON data transformation."""
    print("Testing JSON transformation...")
    
    # Parse JSON data
    parser = DataParser()
    parse_result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.json")
    
    if not parse_result.success:
        print(f"✗ JSON parsing failed: {parse_result.error}")
        return
    
    # Transform parsed data
    transformer = DataTransformer()
    transform_result = transformer.transform_from_dataframe(parse_result.data, "json")
    
    if transform_result.success:
        print(f"✓ JSON transformation successful: {type(transform_result.data).__name__}")
        print(f"  Records transformed: {len(transform_result.data)}")
        print(f"  Sample record: {transform_result.data[0]}")
    else:
        print(f"✗ JSON transformation failed: {transform_result.error}")
    print()

def test_xml_transformation():
    """Test XML data transformation."""
    print("Testing XML transformation...")
    
    # Parse XML data
    parser = DataParser()
    parse_result = parser.parse("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.xml")
    
    if not parse_result.success:
        print(f"✗ XML parsing failed: {parse_result.error}")
        return
    
    # Transform parsed data (XML structure is different)
    transformer = DataTransformer()
    transform_result = transformer.transform_from_dict(parse_result.data, "xml")
    
    if transform_result.success:
        print(f"✓ XML transformation successful: {type(transform_result.data).__name__}")
        print(f"  Records transformed: {len(transform_result.data)}")
        print(f"  Sample record: {transform_result.data[0]}")
    else:
        print(f"✗ XML transformation failed: {transform_result.error}")
    print()

def test_data_validation():
    """Test data validation functionality."""
    print("Testing data validation...")
    
    transformer = DataTransformer()
    
    # Test valid data
    valid_data = {
        "name": "Test User",
        "age": 30,
        "city": "Test City",
        "score": 85.5
    }
    
    result = transformer.validate_data(valid_data, "test")
    print(f"Valid data validation: {'✓' if result.success else '✗'} {result.error if not result.success else ''}")
    
    # Test invalid data (missing required field)
    invalid_data = {
        "name": "Test User",
        "age": 30,
        "city": "Test City"
        # Missing score
    }
    
    result = transformer.validate_data(invalid_data, "test")
    print(f"Invalid data validation: {'✓' if not result.success else '✗'} {result.error}")
    
    # Test invalid data (wrong type)
    wrong_type_data = {
        "name": "Test User",
        "age": "not_a_number",  # Should be int
        "city": "Test City",
        "score": 85.5
    }
    
    result = transformer.validate_data(wrong_type_data, "test")
    print(f"Wrong type validation: {'✓' if not result.success else '✗'} {result.error}")
    print()

def test_schema_compatibility():
    """Test schema compatibility checking."""
    print("Testing schema compatibility...")
    
    transformer = DataTransformer()
    
    # Test compatible data
    compatible_data = [
        {"name": "John", "age": 25, "city": "New York", "score": 85.5}
    ]
    
    result = transformer.validate_schema_compatibility(compatible_data)
    print(f"Compatible schema check: {'✓' if result.success else '✗'} {result.data.get('message') if result.success else result.error}")
    
    # Test incompatible data (missing fields)
    incompatible_data = [
        {"name": "John", "age": 25}  # Missing city and score
    ]
    
    result = transformer.validate_schema_compatibility(incompatible_data)
    print(f"Incompatible schema check: {'✓' if not result.success else '✗'} {result.error}")
    print()

def test_data_normalization():
    """Test data type normalization."""
    print("Testing data normalization...")
    
    transformer = DataTransformer()
    
    # Test data with mixed types
    mixed_type_data = {
        "name": "Test User",
        "age": "30",  # String that should be int
        "city": "Test City",
        "score": "85.5"  # String that should be float
    }
    
    result = transformer.normalize_data_types(mixed_type_data)
    
    if result.success:
        print(f"✓ Data normalization successful")
        print(f"  Normalized data: {result.data}")
        print(f"  Age type: {type(result.data['age']).__name__}")
        print(f"  Score type: {type(result.data['score']).__name__}")
    else:
        print(f"✗ Data normalization failed: {result.error}")
    print()

def test_error_handling():
    """Test error handling in transformation."""
    print("Testing error handling...")
    
    transformer = DataTransformer()
    
    # Test with invalid data structure
    invalid_data = "not_a_dict_or_list"
    
    result = transformer.validate_data(invalid_data, "test")
    print(f"Invalid data structure: {'✓' if not result.success else '✗'} {result.error}")
    
    # Test with empty data
    empty_data = []
    
    result = transformer.validate_data(empty_data, "test")
    print(f"Empty data: {'✓' if result.success else '✗'} {len(result.data) if result.success else result.error}")
    print()

def test_schema_inspection():
    """Test schema inspection functionality."""
    print("Testing schema inspection...")
    
    transformer = DataTransformer()
    schema = transformer.get_schema()
    
    print(f"✓ Schema retrieved successfully")
    print(f"  Schema fields: {list(schema['properties'].keys())}")
    print(f"  Required fields: {schema.get('required', [])}")
    print()

def main():
    """Run all transformation tests."""
    print("=" * 60)
    print("DATA TRANSFORMATION COMPONENT TEST")
    print("=" * 60)
    print()
    
    try:
        test_csv_transformation()
        test_json_transformation()
        test_xml_transformation()
        test_data_validation()
        test_schema_compatibility()
        test_data_normalization()
        test_error_handling()
        test_schema_inspection()
        
        print("=" * 60)
        print("TRANSFORMATION TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()