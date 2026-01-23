#!/usr/bin/env python3

"""
Test script for the dataset parser component.
"""

import sys
import os
import asyncio

# Add the project root to Python path
sys.path.insert(0, "/Users/dev/Documents/GitHub/atlastrinity")

from etl_module.src.parsing import DatasetParser

def test_company_dataset():
    """Test parsing of company dataset."""
    print("Testing company dataset parsing...")
    
    # Create a test company dataset
    test_company_data = [
        {
            "company": "Tech Corp",
            "registration": "12345678",
            "address": "123 Tech Street, Silicon Valley",
            "director": "John Smith"
        },
        {
            "name": "Innovate Ltd",
            "edrpou": "87654321",
            "location": "456 Innovation Ave, San Francisco",
            "directors": ["Alice Johnson", "Bob Williams"]
        }
    ]
    
    # Save test data
    import json
    test_file = "/Users/dev/Documents/GitHub/atlastrinity/test_data/test_companies.json"
    with open(test_file, 'w') as f:
        json.dump(test_company_data, f, indent=2)
    
    # Parse the dataset
    parser = DatasetParser()
    result = parser.parse_dataset(test_file, dataset_type="company")
    
    if result["success"]:
        print(f"✓ Company dataset parsing successful")
        print(f"  Dataset type: {result['dataset_type']}")
        print(f"  Records extracted: {len(result['transformed_data'])}")
        print(f"  Sample data: {result['transformed_data'][0]}")
        return result
    else:
        print(f"✗ Company dataset parsing failed: {result['error']}")
        return None

def test_director_dataset():
    """Test parsing of director dataset."""
    print("\nTesting director dataset parsing...")
    
    # Create a test director dataset
    test_director_data = [
        {
            "name": "John Smith",
            "position": "CEO",
            "company": "Tech Corp"
        },
        {
            "director": "Alice Johnson",
            "role": "CTO",
            "organization": "Innovate Ltd"
        }
    ]
    
    # Save test data
    import json
    test_file = "/Users/dev/Documents/GitHub/atlastrinity/test_data/test_directors.json"
    with open(test_file, 'w') as f:
        json.dump(test_director_data, f, indent=2)
    
    # Parse the dataset
    parser = DatasetParser()
    result = parser.parse_dataset(test_file, dataset_type="director")
    
    if result["success"]:
        print(f"✓ Director dataset parsing successful")
        print(f"  Dataset type: {result['dataset_type']}")
        print(f"  Records extracted: {len(result['transformed_data'])}")
        print(f"  Sample data: {result['transformed_data'][0]}")
        return result
    else:
        print(f"✗ Director dataset parsing failed: {result['error']}")
        return None

def test_auto_detection():
    """Test auto-detection of dataset type."""
    print("\nTesting auto-detection...")
    
    parser = DatasetParser()
    
    # Test with existing test files
    test_files = [
        "/Users/dev/Documents/GitHub/atlastrinity/test_data/test.csv",
        "/Users/dev/Documents/GitHub/atlastrinity/test_data/test.json",
        "/Users/dev/Documents/GitHub/atlastrinity/test_data/test.xml"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            result = parser.parse_dataset(test_file, dataset_type="auto")
            if result["success"]:
                print(f"✓ {test_file}: Auto-detected as {result['dataset_type']}")
            else:
                print(f"✗ {test_file}: Failed - {result['error']}")

def test_knowledge_graph_logging():
    """Test Knowledge Graph logging functionality."""
    print("\nTesting Knowledge Graph logging...")
    
    # First get a successful parsing result
    parser = DatasetParser()
    result = parser.parse_dataset("/Users/dev/Documents/GitHub/atlastrinity/test_data/test.json")
    
    if result and result["success"]:
        # Test KG logging
        try:
            kg_result = asyncio.run(parser.log_to_knowledge_graph(result))
            if kg_result["success"]:
                print(f"✓ Knowledge Graph logging successful")
                print(f"  Nodes added: {kg_result['nodes_added']}")
                print(f"  Edges added: {kg_result['edges_added']}")
            else:
                print(f"✗ Knowledge Graph logging failed: {kg_result['error']}")
        except Exception as e:
            print(f"✗ Knowledge Graph logging exception: {str(e)}")
    else:
        print("✗ Skipping KG test - no successful parsing result")

def main():
    """Run all tests."""
    print("=" * 60)
    print("DATASET PARSER COMPONENT TEST")
    print("=" * 60)
    print()
    
    try:
        company_result = test_company_dataset()
        director_result = test_director_dataset()
        test_auto_detection()
        test_knowledge_graph_logging()
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
        # Return success if at least one test passed
        if company_result or director_result:
            return 0
        else:
            return 1
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())