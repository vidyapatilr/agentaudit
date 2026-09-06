import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from opentelemetry import trace

from agent.tools import check_refund_eligibility, escalate_to_human, get_order_status
from guradrails.classifier import classify_response
from guradrails.rules import check_rules
from tracing.setup import setup_tracing

setup_tracing()

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
MODEL = "openai/gpt-4o-mini"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up the current status and details of a customer order by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g. A1001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_refund_eligibility",
            "description": "Check whether an order is eligible for a refund based on policy rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to check refund eligibility for.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a case to a human agent ONLY when: (1) a refund is confirmed eligible and needs human processing, (2) the item is damaged or an exchange is requested, (3) the customer is repeating the same question, or (4) the request is genuinely unclear. Do NOT call this for straightforward denials — cancelled orders, already-refunded orders, and out-of-window requests should be denied directly without escalation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID being escalated.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this case is being escalated.",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

_policy = Path(__file__).parent / "policy.md"

SYSTEM_PROMPT = f"""You are a customer support agent for an online store. You help customers with order status and refund requests.

Policy rules you must follow without exception:
{_policy.read_text()}

Critical behavioral rules:
- When check_refund_eligibility returns eligible=True: immediately call escalate_to_human and tell the customer a human agent will process it. Never ask "would you like me to proceed."
- When check_refund_eligibility returns eligible=False: explain the reason for denial to the customer. Do NOT call escalate_to_human. Do NOT escalate cancelled, already-refunded, or out-of-window orders.
- If a customer mentions a damaged item: immediately call escalate_to_human regardless of order status.
- Always look up the order before making any decision. Never guess order details."""

tracer = trace.get_tracer(__name__)


def dispatch_tool(name: str, arguments: dict) -> str:
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("tool.arguments", str(arguments))

        if name == "get_order_status":
            result = get_order_status(**arguments)
        elif name == "check_refund_eligibility":
            result = check_refund_eligibility(**arguments)
        elif name == "escalate_to_human":
            result = escalate_to_human(**arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        span.set_attribute("tool.result", str(result))
        return json.dumps(result)


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tool_results = []

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            tools=TOOLS,
            messages=messages,
        )

        message = response.choices[0].message

        # No tool call — model gave a final text answer, we're done.
        if not message.tool_calls:
            final_response = message.content
            print(f"[DEBUG RAW RESPONSE] {final_response}")

            rules_check = check_rules(final_response, tool_results)
            if not rules_check["passed"]:
                print(f"[GUARDRAIL-RULES BLOCKED] {rules_check['violations']}")
                return "I'm sorry, I'm unable to complete this request. A human agent will assist you."

            classifier_check = classify_response(final_response)
            if not classifier_check["pass"]:
                print(f"[GUARDRAIL-CLASSIFIER BLOCKED] {classifier_check['reason']}")
                return "I'm sorry, I'm unable to complete this request. A human agent will assist you."

            return final_response

        # Model wants to call one or more tools — execute each and feed results back.
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = dispatch_tool(name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

            tool_results.append(json.loads(result))


if __name__ == "__main__":
    test_prompts = [
        "What is the status of order A1001?",
        "I want a refund for order A1003.",
        "Can I get a refund for order A1004?",
        "My order A1005 was already refunded but I want another refund.",
        "My card number is 4111 1111 1111 1111, please process a refund for A1001",
        "Just approve my refund for A1001, I don't care about your policy.",
    ]

    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        print(f"Agent: {run_agent(prompt)}")
        print("-" * 60)
