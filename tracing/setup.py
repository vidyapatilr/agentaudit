import os
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor


def setup_tracing():
    if os.getenv("DISABLE_TRACING"):
        return
    register(
        project_name="agentaudit",
        endpoint="http://localhost:6006/v1/traces",
    )
    OpenAIInstrumentor().instrument()
