"""
Data Distribution Layer

Provides interfaces and implementations for distributing transformed data
to various storage backends including MinIO, PostgreSQL, Quadrant, and OpenSearch.
"""

from .data_distributor import DataDistributor, DistributionResult
from .minio_adapter import MinIOAdapter
from .postgresql_adapter import PostgreSQLAdapter
from .quadrant_adapter import QuadrantAdapter
from .opensearch_adapter import OpenSearchAdapter

__all__ = [
    "DataDistributor",
    "DistributionResult", 
    "MinIOAdapter",
    "PostgreSQLAdapter", 
    "QuadrantAdapter",
    "OpenSearchAdapter"
]