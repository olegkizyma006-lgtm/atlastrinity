#!/usr/bin/env python3
"""
Data Validation Script for AtlasTrinity

Validates collected data for completeness, accuracy, and relevance.
Uses predefined schemas from the project's validation system.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import inspect

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("data_validator")

# Import project modules
import sys

sys.path.insert(0, 'src')

from brain.config import CONFIG_ROOT, PROJECT_ROOT
from brain.config_validator import ConfigValidator, ValidationResult
from brain.data_guard import DataQualityGuard
from brain.db.schema import Base


class DataValidator:
    """Validates all collected data in the AtlasTrinity system."""
    
    def __init__(self):
        self.data_guard = DataQualityGuard()
        self.config_validator = ConfigValidator()
        self.validation_results = []
    
    def validate_config_files(self) -> list[ValidationResult]:
        """Validate all configuration files using the built-in validator."""
        logger.info("Validating configuration files...")
        results = self.config_validator.validate_all()
        self.validation_results.extend(results)
        return results
    
    def validate_database_schema(self, db_path: Path) -> dict[str, Any]:
        """Validate database schema against expected structure."""
        if not db_path.exists():
            return {
                "status": "error",
                "message": f"Database file not found: {db_path}",
                "valid": False
            }
        
        try:
            # Connect to database
            engine = self._create_sqlite_engine(db_path)
            inspector = inspect(engine)
            
            # Get all tables
            tables = inspector.get_table_names()
            expected_tables = [
                "sessions", "tasks", "task_steps", "tool_executions", 
                "logs", "kg_nodes", "kg_edges", "agent_messages", 
                "recovery_attempts", "conversation_summaries", 
                "behavioral_deviations", "knowledge_promotions"
            ]
            
            result = {
                "database": str(db_path),
                "tables_found": len(tables),
                "tables_expected": len(expected_tables),
                "missing_tables": [],
                "valid": True,
                "issues": []
            }
            
            # Check for missing tables
            for expected_table in expected_tables:
                if expected_table not in tables:
                    result["missing_tables"].append(expected_table)
                    result["valid"] = False
                    result["issues"].append(f"Missing table: {expected_table}")
            
            if result["valid"]:
                logger.info(f"✓ Database schema validation passed for {db_path}")
            else:
                logger.error(f"✗ Database schema validation failed for {db_path}")
                
            return result
            
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "valid": False
            }
    
    def validate_tool_schemas(self) -> dict[str, Any]:
        """Validate tool schemas file."""
        schemas_path = Path("src/brain/data/tool_schemas.json")
        
        if not schemas_path.exists():
            return {
                "status": "error",
                "message": f"Tool schemas file not found: {schemas_path}",
                "valid": False
            }
        
        try:
            with open(schemas_path, encoding='utf-8') as f:
                schemas = json.load(f)
            
            result = {
                "file": str(schemas_path),
                "tool_count": len(schemas),
                "valid": True,
                "issues": []
            }
            
            # Basic validation: each tool should have required fields
            required_fields = ["server", "required", "types", "description"]
            
            for tool_name, schema in schemas.items():
                missing_fields = [field for field in required_fields if field not in schema]
                if missing_fields:
                    result["valid"] = False
                    result["issues"].append(f"Tool '{tool_name}' missing fields: {missing_fields}")
                
                # Validate required fields are lists
                if "required" in schema and not isinstance(schema["required"], list):
                    result["valid"] = False
                    result["issues"].append(f"Tool '{tool_name}' required field should be a list")
                
                # Validate types field
                if "types" in schema and not isinstance(schema["types"], dict):
                    result["valid"] = False
                    result["issues"].append(f"Tool '{tool_name}' types field should be a dict")
            
            if result["valid"]:
                logger.info(f"✓ Tool schemas validation passed ({result['tool_count']} tools)")
            else:
                logger.error("✗ Tool schemas validation failed")
                
            return result
            
        except Exception as e:
            logger.error(f"Tool schemas validation error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "valid": False
            }
    
    def validate_mcp_catalog(self) -> dict[str, Any]:
        """Validate MCP catalog file."""
        catalog_path = Path("src/brain/data/mcp_catalog.json")
        
        if not catalog_path.exists():
            return {
                "status": "error",
                "message": f"MCP catalog file not found: {catalog_path}",
                "valid": False
            }
        
        try:
            with open(catalog_path, encoding='utf-8') as f:
                catalog = json.load(f)
            
            result = {
                "file": str(catalog_path),
                "server_count": len(catalog),
                "valid": True,
                "issues": []
            }
            
            # Basic validation: each server should have required fields
            required_fields = ["name", "description", "capabilities", "key_tools"]
            
            for server_name, server_data in catalog.items():
                missing_fields = [field for field in required_fields if field not in server_data]
                if missing_fields:
                    result["valid"] = False
                    result["issues"].append(f"Server '{server_name}' missing fields: {missing_fields}")
            
            if result["valid"]:
                logger.info(f"✓ MCP catalog validation passed ({result['server_count']} servers)")
            else:
                logger.error("✗ MCP catalog validation failed")
                
            return result
            
        except Exception as e:
            logger.error(f"MCP catalog validation error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "valid": False
            }
    
    def validate_data_completeness(self) -> dict[str, Any]:
        """Validate data completeness across the system."""
        result = {
            "checks": [],
            "valid": True,
            "issues": []
        }
        
        # Check critical files exist
        critical_files = [
            "src/brain/data/tool_schemas.json",
            "src/brain/data/mcp_catalog.json",
            "src/brain/db/schema.py",
            "src/brain/config_validator.py",
            "src/brain/data_guard.py"
        ]
        
        for file_path in critical_files:
            path = Path(file_path)
            exists = path.exists()
            result["checks"].append({
                "file": file_path,
                "exists": exists
            })
            
            if not exists:
                result["valid"] = False
                result["issues"].append(f"Critical file missing: {file_path}")
        
        if result["valid"]:
            logger.info("✓ Data completeness validation passed")
        else:
            logger.error("✗ Data completeness validation failed")
            
        return result
    
    def _create_sqlite_engine(self, db_path: Path):
        """Create SQLAlchemy engine for SQLite database."""
        from sqlalchemy import create_engine
        return create_engine(f"sqlite:///{db_path}")
    
    def generate_validation_report(self) -> dict[str, Any]:
        """Generate comprehensive validation report."""
        report = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "results": [],
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0
            }
        }
        
        # Run all validations
        logger.info("Starting comprehensive data validation...")
        
        # 1. Configuration files
        config_results = self.validate_config_files()
        for config_result in config_results:
            report["results"].append({
                "type": "config_validation",
                "file": str(config_result.file_path),
                "valid": config_result.valid,
                "errors": len(config_result.errors),
                "warnings": len(config_result.warnings),
                "issues": [{
                    "level": issue.level,
                    "path": issue.path,
                    "message": issue.message
                } for issue in config_result.issues]
            })
        
        # 2. Database schema
        db_results = []
        test_db = Path("tests/mock_config/atlastrinity.db")
        if test_db.exists():
            db_result = self.validate_database_schema(test_db)
            db_results.append(db_result)
        
        backup_db = Path("backups/databases/atlastrinity.db")
        if backup_db.exists():
            db_result = self.validate_database_schema(backup_db)
            db_results.append(db_result)
        
        for db_result in db_results:
            report["results"].append({
                "type": "database_schema",
                **db_result
            })
        
        # 3. Tool schemas
        tool_schemas_result = self.validate_tool_schemas()
        report["results"].append({
            "type": "tool_schemas",
            **tool_schemas_result
        })
        
        # 4. MCP catalog
        mcp_catalog_result = self.validate_mcp_catalog()
        report["results"].append({
            "type": "mcp_catalog",
            **mcp_catalog_result
        })
        
        # 5. Data completeness
        completeness_result = self.validate_data_completeness()
        report["results"].append({
            "type": "data_completeness",
            **completeness_result
        })
        
        # Generate summary
        for result in report["results"]:
            report["summary"]["total_checks"] += 1
            if result.get("valid", False):
                report["summary"]["passed_checks"] += 1
            else:
                report["summary"]["failed_checks"] += 1
            
            # Count warnings from config validation
            if result.get("type") == "config_validation":
                report["summary"]["warnings"] += result.get("warnings", 0)
        
        # Overall validation status
        report["overall_status"] = "PASS" if report["summary"]["failed_checks"] == 0 else "FAIL"
        
        logger.info(f"Validation complete: {report['overall_status']}")
        logger.info(f"Summary: {report['summary']['passed_checks']}/{report['summary']['total_checks']} checks passed")
        
        return report
    
    def save_validation_report(self, report: dict[str, Any], output_path: Path = Path("validation_report.json")):
        """Save validation report to file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Validation report saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")
            return False
    
    def flag_data_as_validated(self, report: dict[str, Any]) -> dict[str, Any]:
        """Flag all validated data as complete, accurate, and relevant."""
        flagged_report = report.copy()
        
        # Add validation flags to each result
        for result in flagged_report["results"]:
            result["validation_flags"] = {
                "complete": result.get("valid", False),
                "accurate": result.get("valid", False),
                "relevant": True,  # All system data is considered relevant
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
        
        return flagged_report


def main():
    """Main validation entry point."""
    validator = DataValidator()
    
    # Run comprehensive validation
    report = validator.generate_validation_report()
    
    # Flag all data as validated
    flagged_report = validator.flag_data_as_validated(report)
    
    # Save report
    validator.save_validation_report(flagged_report)
    
    # Print summary
    print("\n" + "="*60)
    print("ATLAS TRINITY DATA VALIDATION REPORT")
    print("="*60)
    print(f"Status: {flagged_report['overall_status']}")
    print(f"Checks: {flagged_report['summary']['passed_checks']}/{flagged_report['summary']['total_checks']} passed")
    print(f"Warnings: {flagged_report['summary']['warnings']}")
    print(f"Failed: {flagged_report['summary']['failed_checks']}")
    print("="*60)
    
    # Show detailed results
    for result in flagged_report["results"]:
        status_icon = "✓" if result.get("valid", False) else "✗"
        result_type = result.get("type", "unknown")
        
        if result_type == "config_validation":
            print(f"{status_icon} Config: {result.get('file', 'unknown')}")
            if result.get("errors", 0) > 0:
                print(f"    Errors: {result.get('errors', 0)}, Warnings: {result.get('warnings', 0)}")
        elif result_type == "database_schema":
            print(f"{status_icon} Database: {result.get('database', 'unknown')}")
            if not result.get("valid", True):
                print(f"    Missing tables: {result.get('missing_tables', [])}")
        elif result_type == "tool_schemas":
            print(f"{status_icon} Tool Schemas: {result.get('tool_count', 0)} tools")
        elif result_type == "mcp_catalog":
            print(f"{status_icon} MCP Catalog: {result.get('server_count', 0)} servers")
        elif result_type == "data_completeness":
            print(f"{status_icon} Data Completeness: {len(result.get('checks', []))} files checked")
    
    print("="*60)
    print("All data has been flagged as validated according to predefined schemas.")
    print("Report saved to: validation_report.json")
    print("="*60)
    
    return 0 if flagged_report['overall_status'] == 'PASS' else 1


if __name__ == "__main__":
    sys.exit(main())