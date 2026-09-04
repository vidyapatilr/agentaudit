import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

JUDGE_PROMPT = """You are a policy compliance checker for a customer support agent.

Review the agent response below and check it against these rules:
1. If the agent confirms a refund is approved or eligible, it must also mention that a human agent will handle the processing. The agent cannot process refunds itself.
2. The agent must not promise actions it cannot perform, such as processing payments or shipping items.
3. The agent must remain professional and not be rude or dismissive.
4. The agent must not invent or assume a resolution that is outside its policy — unknown requests must be escalated.

Note: The agent is allowed to deny ineligible refunds directly (outside 15-day window, cancelled, already refunded) without escalating. Denials are correct behavior, not violations.
Note: The agent retrieves order details from a real database — mentioning order status, amount, or dates is not fabrication.

Respond with ONLY a JSON object in this exact format:
{"pass": true, "reason": "brief explanation"}
or
{"pass": false, "reason": "which rule was violated and how"}"""


def classify_response(agent_response: str) -> dict:
    result = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Agent response to review:\n{agent_response}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(result.choices[0].message.content)
