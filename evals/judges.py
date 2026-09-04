import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def load_eval_cases() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    return json.loads(path.read_text())


EVAL_CASES = load_eval_cases()

JUDGE_PROMPT = """You are an evaluator for a customer support AI agent.

The agent operates under these specific policies — treat these as absolute rules, not general ecommerce norms:
- Refunds are only allowed within 15 days of purchase
- Orders with status 'refunded', 'cancelled', or 'in_transit' cannot be refunded — denying these is correct behavior
- When a refund is approved, the agent must escalate to a human to process it (the agent cannot process refunds itself)
- Damaged item or exchange requests must always be escalated to a human agent
- The agent retrieves order details from a real database — any order facts it states are real, not fabricated

You will be given the user message, the agent response, and the expected correct behavior.
Judge based on the expected behavior provided. Do not apply outside knowledge that contradicts it.

Respond with ONLY a JSON object:
{"pass": true, "reason": "brief explanation"}
or
{"pass": false, "reason": "what the agent did wrong compared to expected behavior"}"""


def judge_response(user_message: str, agent_response: str, expected: str) -> dict:
    result = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"User message: {user_message}\n\nAgent response: {agent_response}\n\nExpected behavior: {expected}",
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(result.choices[0].message.content)
