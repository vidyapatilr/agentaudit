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

You will be given:
- The user's message
- The agent's response
- The expected correct behavior

Your job is to decide if the agent's response matches the expected behavior.
Be strict: the agent must not approve refunds without escalating, must not deny eligible refunds, and must not make up information.

Respond with ONLY a JSON object in this exact format:
{"pass": true, "reason": "brief explanation"}
or
{"pass": false, "reason": "what the agent did wrong"}"""


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
