"""
OpenTelemetry configuration for the Image Styler application.
Sets up tracing and metrics with OTLP exporters.
"""
import os
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from logger_config import setup_logger

logger = setup_logger()

# Service information
SERVICE_INFO = {
    SERVICE_NAME: "image-styler",
    SERVICE_VERSION: "0.1.0",
}

# OTLP Collector endpoint (from docker-compose or env var)
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")


def setup_telemetry():
    """
    Initialize OpenTelemetry tracing and metrics.
    Call this once at application startup.
    """
    try:
        # Create resource with service information
        resource = Resource.create(SERVICE_INFO)
        
        # ===== TRACING SETUP =====
        trace_provider = TracerProvider(resource=resource)
        
        # Configure OTLP exporter for traces
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True  # For local development
        )
        
        # Add batch span processor
        trace_provider.add_span_processor(
            BatchSpanProcessor(otlp_trace_exporter)
        )
        
        # Set the global tracer provider
        trace.set_tracer_provider(trace_provider)
        
        # ===== METRICS SETUP =====
        # Configure OTLP exporter for metrics
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=OTLP_ENDPOINT,
            insecure=True
        )
        
        # Create metric reader with periodic export
        metric_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=15000  # Export every 15 seconds
        )
        
        # Set the global meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        
        logger.info(f"OpenTelemetry initialized. Exporting to {OTLP_ENDPOINT}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
        return False


def get_tracer(name: str):
    """Get a tracer instance for creating spans."""
    return trace.get_tracer(name, "0.1.0")


def get_meter(name: str):
    """Get a meter instance for creating metrics."""
    return metrics.get_meter(name, "0.1.0")


# Create global tracer and meter instances
tracer = get_tracer("image-styler")
meter = get_meter("image-styler")

# ===== METRICS DEFINITIONS =====
# Counters
images_processed_counter = meter.create_counter(
    name="images_processed_total",
    description="Total number of images processed",
    unit="1"
)

api_calls_counter = meter.create_counter(
    name="api_calls_total",
    description="Total number of API calls made",
    unit="1"
)

api_errors_counter = meter.create_counter(
    name="api_errors_total",
    description="Total number of API errors",
    unit="1"
)

# Histograms
api_duration_histogram = meter.create_histogram(
    name="api_call_duration_seconds",
    description="API call duration in seconds",
    unit="s"
)

request_duration_histogram = meter.create_histogram(
    name="request_duration_seconds",
    description="End-to-end request duration in seconds",
    unit="s"
)
