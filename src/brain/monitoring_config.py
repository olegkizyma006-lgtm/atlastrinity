"""
Monitoring Configuration Loader

Loads and manages monitoring configuration for Prometheus, Grafana, and OpenSearch integration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringConfig:
    """
    Monitoring configuration loader and manager.
    
    This class handles loading monitoring configuration from YAML files
    and provides access to monitoring settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the monitoring configuration loader.
        
        Args:
            config_path: Optional path to monitoring config file
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> Path:
        """
        Get the default monitoring configuration path.
        
        Returns:
            Path to the default monitoring configuration file
        """
        # Check for config in standard locations
        possible_paths = [
            Path("/etc/atlastrinity/monitoring_config.yaml"),
            Path.home() / ".config" / "atlastrinity" / "monitoring_config.yaml",
            Path("config/monitoring_config.yaml"),
            Path("monitoring_config.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # Return default location (will be created if needed)
        return Path.home() / ".config" / "atlastrinity" / "monitoring_config.yaml"
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load monitoring configuration from YAML file.
        
        Returns:
            Dictionary containing monitoring configuration
        """
        default_config = self._get_default_config()
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
                    
                # Deep merge user config with defaults
                return self._deep_merge(default_config, user_config)
            else:
                logger.info(f"Monitoring config file not found at {self.config_path}, using defaults")
                return default_config
                
        except Exception as e:
            logger.error(f"Error loading monitoring config: {e}")
            logger.info("Falling back to default configuration")
            return default_config
            
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default monitoring configuration.
        
        Returns:
            Dictionary containing default monitoring configuration
        """
        return {
            "monitoring": {
                "prometheus": {
                    "enabled": True,
                    "port": 8001,
                    "scrape_interval": "15s",
                    "evaluation_interval": "15s"
                },
                "grafana": {
                    "enabled": True,
                    "logging_format": "json",
                    "log_level": "info"
                },
                "opensearch": {
                    "enabled": True,
                    "hosts": [],
                    "index_prefix": "atlastrinity"
                },
                "tracing": {
                    "enabled": True,
                    "service_name": "atlastrinity",
                    "otlp_endpoint": "localhost:4317",
                    "batch_timeout": "5s",
                    "max_export_batch_size": 512
                },
                "etl": {
                    "enabled": True,
                    "track_stages": ["scraping", "transformation", "distribution", "indexing"],
                    "error_thresholds": {
                        "warning": 5,
                        "critical": 10
                    }
                },
                "alerts": {
                    "high_cpu_usage": {
                        "threshold": 90,
                        "duration": "5m",
                        "severity": "warning"
                    },
                    "high_memory_usage": {
                        "threshold": 85,
                        "duration": "5m",
                        "severity": "warning"
                    },
                    "request_failures": {
                        "threshold": 10,
                        "duration": "1m",
                        "severity": "critical"
                    }
                }
            }
        }
        
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            overlay: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
        
    def get_prometheus_config(self) -> Dict[str, Any]:
        """
        Get Prometheus configuration.
        
        Returns:
            Prometheus configuration dictionary
        """
        return self.config.get("monitoring", {}).get("prometheus", {})
        
    def get_grafana_config(self) -> Dict[str, Any]:
        """
        Get Grafana configuration.
        
        Returns:
            Grafana configuration dictionary
        """
        return self.config.get("monitoring", {}).get("grafana", {})
        
    def get_opensearch_config(self) -> Dict[str, Any]:
        """
        Get OpenSearch configuration.
        
        Returns:
            OpenSearch configuration dictionary
        """
        return self.config.get("monitoring", {}).get("opensearch", {})
        
    def get_tracing_config(self) -> Dict[str, Any]:
        """
        Get tracing configuration.
        
        Returns:
            Tracing configuration dictionary
        """
        return self.config.get("monitoring", {}).get("tracing", {})
        
    def get_etl_config(self) -> Dict[str, Any]:
        """
        Get ETL monitoring configuration.
        
        Returns:
            ETL configuration dictionary
        """
        return self.config.get("monitoring", {}).get("etl", {})
        
    def get_alerts_config(self) -> Dict[str, Any]:
        """
        Get alerts configuration.
        
        Returns:
            Alerts configuration dictionary
        """
        return self.config.get("monitoring", {}).get("alerts", {})
        
    def is_prometheus_enabled(self) -> bool:
        """
        Check if Prometheus monitoring is enabled.
        
        Returns:
            True if Prometheus is enabled, False otherwise
        """
        return self.get_prometheus_config().get("enabled", True)
        
    def is_grafana_enabled(self) -> bool:
        """
        Check if Grafana logging is enabled.
        
        Returns:
            True if Grafana is enabled, False otherwise
        """
        return self.get_grafana_config().get("enabled", True)
        
    def is_opensearch_enabled(self) -> bool:
        """
        Check if OpenSearch integration is enabled.
        
        Returns:
            True if OpenSearch is enabled, False otherwise
        """
        return self.get_opensearch_config().get("enabled", True)
        
    def is_tracing_enabled(self) -> bool:
        """
        Check if tracing is enabled.
        
        Returns:
            True if tracing is enabled, False otherwise
        """
        return self.get_tracing_config().get("enabled", True)
        
    def get_config_path(self) -> Path:
        """
        Get the path to the monitoring configuration file.
        
        Returns:
            Path to the configuration file
        """
        return self.config_path
        
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save monitoring configuration to file.
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False)
                
            logger.info(f"Monitoring configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving monitoring config: {e}")
            return False


# Global monitoring config instance
monitoring_config = MonitoringConfig()