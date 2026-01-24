"""
Blob Storage Adapter for Golden Fund
Ported from etl_module/src/distribution/minio_adapter.py
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import logging
from pathlib import Path
import uuid
from .types import StorageResult

logger = logging.getLogger("golden_fund.storage.blob")

class BlobStorage:
    """
    Blob storage adapter (MinIO-style).
    Persists data to local disk in a structured way (simulating bucket storage).
    """
    
    def __init__(self, root_path: str = None, bucket: str = "default"):
        if root_path is None:
            root_path = Path.home() / ".config" / "atlastrinity" / "data" / "golden_fund" / "blobs"
        self.root = Path(root_path)
        self.bucket = bucket
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket_path = self.root / self.bucket
        self.bucket_path.mkdir(exist_ok=True)
        logger.info(f"BlobStorage initialized at {self.bucket_path}")

    def store(self, data: Any, filename: Optional[str] = None) -> StorageResult:
        try:
            if not filename:
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            
            file_path = self.bucket_path / filename
            
            with open(file_path, "w", encoding="utf-8") as f:
                if isinstance(data, (dict, list)):
                    json.dump(data, f, indent=2, default=str)
                else:
                    f.write(str(data))
            
            logger.info(f"Stored blob: {file_path}")
            
            return StorageResult(True, "blob", data={
                "path": str(file_path),
                "filename": filename,
                "url": f"file://{file_path.absolute()}", # Simulated URL
                "size": file_path.stat().st_size
            })
            
        except Exception as e:
            return StorageResult(False, "blob", error=str(e))

    def retrieve(self, filename: str) -> StorageResult:
        try:
            file_path = self.bucket_path / filename
            if not file_path.exists():
                return StorageResult(False, "blob", error="File not found")
                
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    data = f.read()
                    
            return StorageResult(True, "blob", data=data)
        except Exception as e:
            return StorageResult(False, "blob", error=str(e))
