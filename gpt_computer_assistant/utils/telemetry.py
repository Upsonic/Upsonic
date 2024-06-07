from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode


def CreateTracer(service_name, trace_name, infrastackai_api_key=None):

    tracer = trace.get_tracer(trace_name)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="https://collector-us1-http.infrastack.ai/v1/traces",
                headers=(("infrastack-api-key", infrastackai_api_key),),
            )
        )
    )

    return tracer


def os_name():
    import platform

    system_name = platform.system()
    if system_name == "Windows":
        return "Windows"
    elif system_name == "Darwin":
        return "macOS"
    elif system_name == "Linux":
        return "Linux"
    else:
        return "Unknown OS"


my_tracer = CreateTracer(
    "gpt_computer_assistant",
    "app",
    infrastackai_api_key="sk-2b29c6da910d2883de0599d4c5dd6b9d2e4ec61bbfa834d5",
)
