from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register


def setup_tracing():
    register(
        project_name="agentaudit",
        endpoint="http://localhost:6006/v1/traces",
    )
    OpenAIInstrumentor().instrument()
