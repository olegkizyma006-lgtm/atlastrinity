"""
AtlasTrinity Monitoring and Logging Integration

Provides comprehensive monitoring with Prometheus, Grafana, and OpenSearch integration
for real-time insights and observability.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Optional, Union

import psutil
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonitoringSystem:
    """
    Comprehensive monitoring system integrating Prometheus, Grafana, and OpenSearch.
    
    This system provides:
    - Real-time metrics collection via Prometheus
    - Distributed tracing with OpenTelemetry
    - Logging integration for Grafana visualization
    - OpenSearch integration for analytics and insights
    """
    
    def __init__(self, 
                 prometheus_port: int = 8001,
                 opensearch_enabled: bool = True,
                 grafana_enabled: bool = True,
                 config: Optional[dict[str, Any]] = None):
        """
        Initialize the monitoring system.
        
        Args:
            prometheus_port: Port for Prometheus metrics server
            opensearch_enabled: Enable OpenSearch integration
            grafana_enabled: Enable Grafana logging integration
            config: Optional monitoring configuration dictionary
        """
        # Load configuration
        self.config = config or self._load_config()
        
        # Apply configuration
        self.prometheus_port = self.config.get("prometheus", {}).get("port", prometheus_port)
        self.opensearch_enabled = self.config.get("opensearch", {}).get("enabled", opensearch_enabled)
        self.grafana_enabled = self.config.get("grafana", {}).get("enabled", grafana_enabled)
        
        # Initialize metrics collectors
        self._initialize_metrics()
        
        # Initialize tracing
        self._initialize_tracing()
        
        # Start Prometheus server
        self._start_prometheus_server()
        
        logger.info(f"Monitoring system initialized - Prometheus on port {self.prometheus_port}")
        
    def _load_config(self) -> dict[str, Any]:
        """
        Load monitoring configuration.
        
        Returns:
            Dictionary containing monitoring configuration
        """
        try:
            from .monitoring_config import monitoring_config
            return {
                "prometheus": monitoring_config.get_prometheus_config(),
                "grafana": monitoring_config.get_grafana_config(),
                "opensearch": monitoring_config.get_opensearch_config(),
                "tracing": monitoring_config.get_tracing_config(),
                "etl": monitoring_config.get_etl_config()
            }
        except ImportError:
            logger.warning("Monitoring config not available, using defaults")
            return {}
        
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics collectors."""
        # System metrics
        self.cpu_usage = Gauge('atlastrinity_cpu_usage_percent', 'Current CPU usage percentage')
        self.memory_usage = Gauge('atlastrinity_memory_usage_bytes', 'Current memory usage in bytes')
        self.disk_usage = Gauge('atlastrinity_disk_usage_bytes', 'Current disk usage in bytes')
        
        # Network metrics
        self.network_bytes_sent = Counter('atlastrinity_network_bytes_sent', 'Total bytes sent')
        self.network_bytes_received = Counter('atlastrinity_network_bytes_received', 'Total bytes received')
        
        # Application metrics
        self.request_count = Counter('atlastrinity_requests_total', 
                                   'Total number of requests processed',
                                   ['request_type', 'status'])
        self.request_latency = Histogram('atlastrinity_request_latency_seconds',
                                       'Request processing latency in seconds',
                                       ['request_type'])
        self.active_requests = Gauge('atlastrinity_active_requests', 'Number of active requests')
        
        # ETL pipeline metrics
        self.etl_records_processed = Counter('atlastrinity_etl_records_processed',
                                           'Number of records processed by ETL',
                                           ['pipeline_stage'])
        self.etl_errors = Counter('atlastrinity_etl_errors',
                                'Number of ETL processing errors',
                                ['pipeline_stage', 'error_type'])
        
        # OpenSearch metrics
        self.opensearch_queries = Counter('atlastrinity_opensearch_queries',
                                        'Number of OpenSearch queries executed',
                                        ['query_type'])
        self.opensearch_documents = Counter('atlastrinity_opensearch_documents',
                                          'Number of documents indexed in OpenSearch')
        
    def _initialize_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        try:
            # Set up resource with service name
            resource = Resource.create({
                "service.name": "atlastrinity",
                "service.version": "1.0.0"
            })
            
            # Create tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Set up span processor with OTLP exporter
            otlp_exporter = OTLPSpanExporter()
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(tracer_provider)
            
            self.tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracing initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
            self.tracer = None
            
    def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.prometheus_port)
            logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            
    def collect_system_metrics(self) -> dict[str, Any]:
        """
        Collect system-level metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(cpu_percent)
            
            # Memory metrics
            mem = psutil.virtual_memory()
            self.memory_usage.set(mem.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.used)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.network_bytes_sent.inc(net_io.bytes_sent)
            self.network_bytes_received.inc(net_io.bytes_recv)
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_used_bytes": mem.used,
                "memory_total_bytes": mem.total,
                "disk_used_bytes": disk.used,
                "disk_total_bytes": disk.total,
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_received": net_io.bytes_recv,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
            
    def record_request(self, request_type: str, status: str, duration: float) -> None:
        """
        Record an application request.
        
        Args:
            request_type: Type of request (e.g., 'chat', 'stt', 'etl')
            status: Status of request (e.g., 'success', 'error')
            duration: Duration in seconds
        """
        try:
            self.request_count.labels(request_type=request_type, status=status).inc()
            self.request_latency.labels(request_type=request_type).observe(duration)
            logger.info(f"Recorded {request_type} request: status={status}, duration={duration:.2f}s")
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
            
    def record_etl_metrics(self, stage: str, records_processed: int, errors: int = 0, error_type: str = "none") -> None:
        """
        Record ETL pipeline metrics.
        
        Args:
            stage: ETL stage (e.g., 'scraping', 'transformation', 'distribution')
            records_processed: Number of records processed
            errors: Number of errors encountered
            error_type: Type of error if any
        """
        try:
            self.etl_records_processed.labels(pipeline_stage=stage).inc(records_processed)
            if errors > 0:
                self.etl_errors.labels(pipeline_stage=stage, error_type=error_type).inc(errors)
            logger.info(f"ETL metrics recorded: stage={stage}, records={records_processed}, errors={errors}")
        except Exception as e:
            logger.error(f"Error recording ETL metrics: {e}")
            
    def record_opensearch_metrics(self, query_type: str, documents: int = 0) -> None:
        """
        Record OpenSearch-related metrics.
        
        Args:
            query_type: Type of OpenSearch operation
            documents: Number of documents involved (for indexing operations)
        """
        try:
            self.opensearch_queries.labels(query_type=query_type).inc()
            if documents > 0:
                self.opensearch_documents.inc(documents)
            logger.info(f"OpenSearch metrics recorded: query_type={query_type}, documents={documents}")
        except Exception as e:
            logger.error(f"Error recording OpenSearch metrics: {e}")
            
    def start_request(self) -> None:
        """Increment active request counter."""
        self.active_requests.inc()
        
    def end_request(self) -> None:
        """Decrement active request counter."""
        self.active_requests.dec()
        
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of current metrics.
        
        Returns:
            Dictionary containing current metrics values
        """
        try:
            return {
                "system": self.collect_system_metrics(),
                "application": {
                    "active_requests": int(self.active_requests._value.get()),
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics snapshot: {e}")
            return {}
            
    def log_for_grafana(self, message: str, level: str = "info", **kwargs) -> None:
        """
        Log message in Grafana-compatible format.
        
        Args:
            message: Log message
            level: Log level (info, warning, error, debug)
            kwargs: Additional context data
        """
        if not self.grafana_enabled:
            return
            
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "service": "atlastrinity",
                **kwargs
            }
            
            log_json = json.dumps(log_entry)
            
            # Log in structured format that Grafana can parse
            if level == "error":
                logger.error(log_json)
            elif level == "warning":
                logger.warning(log_json)
            elif level == "debug":
                logger.debug(log_json)
            else:
                logger.info(log_json)
                
        except Exception as e:
            logger.error(f"Error logging for Grafana: {e}")
            
    def create_span(self, name: str, **kwargs) -> Any:
        """
        Create a tracing span for distributed tracing.
        
        Args:
            name: Span name
            kwargs: Additional span attributes
            
        Returns:
            Tracing span object or None if tracing is disabled
        """
        if not self.tracer:
            return None
            
        try:
            return self.tracer.start_span(name, **kwargs)
        except Exception as e:
            logger.error(f"Error creating tracing span: {e}")
            return None
            
    def is_healthy(self) -> bool:
        """
        Check if monitoring system is healthy.
        
        Returns:
            True if monitoring system is operational, False otherwise
        """
        try:
            # Check if we can collect basic metrics
            metrics = self.collect_system_metrics()
            return bool(metrics)
        except Exception:
            return False


# Global monitoring instance
monitoring_system = MonitoringSystem()