import time
import logging
from typing import Any, Dict, Optional

import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Legacy metrics collector for backward compatibility.
    
    This class provides basic system metrics collection and integrates
    with the new MonitoringSystem for comprehensive observability.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self._last_net_io = psutil.net_io_counters()
        self._last_net_time = time.time()

    def _format_speed(self, bytes_per_sec):
        """Format network speed in human-readable units."""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.0f}", "B/S"
        elif bytes_per_sec < 1024**2:
            return f"{bytes_per_sec / 1024:.1f}", "K/S"
        elif bytes_per_sec < 1024**3:
            return f"{bytes_per_sec / 1024**2:.1f}", "M/S"
        else:
            return f"{bytes_per_sec / 1024**3:.1f}", "G/S"

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get formatted system metrics.
        
        Returns:
            Dictionary containing formatted system metrics
        """
        # CPU
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)

        # Network
        current_net_io = psutil.net_io_counters()
        current_time = time.time()
        time_delta = current_time - self._last_net_time

        if time_delta <= 0:
            time_delta = 0.1  # Prevent div by zero

        bytes_sent = (current_net_io.bytes_sent - self._last_net_io.bytes_sent) / time_delta
        bytes_recv = (current_net_io.bytes_recv - self._last_net_io.bytes_recv) / time_delta

        self._last_net_io = current_net_io
        self._last_net_time = current_time

        up_val, up_unit = self._format_speed(bytes_sent)
        down_val, down_unit = self._format_speed(bytes_recv)

        return {
            "cpu": f"{int(cpu_percent)}%",
            "memory": f"{mem_used_gb:.1f}GB",
            "net_up_val": up_val,
            "net_up_unit": up_unit,
            "net_down_val": down_val,
            "net_down_unit": down_unit,
        }

    def record(self, name: str, value: Any, tags: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a custom metric and integrate with monitoring system.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        try:
            # Log the custom metric
            metric_data = {"name": name, "value": value}
            if tags:
                metric_data["tags"] = tags
            
            logger.info(f"Custom metric recorded: {metric_data}")
            
            # Integrate with monitoring system if available
            try:
                from .monitoring import monitoring_system
                monitoring_system.log_for_grafana(
                    f"Custom metric: {name} = {value}",
                    level="info",
                    metric_name=name,
                    metric_value=value,
                    tags=tags or {}
                )
            except ImportError:
                # Monitoring system not available, continue without it
                pass
                
        except Exception as e:
            logger.error(f"Error recording custom metric: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()
