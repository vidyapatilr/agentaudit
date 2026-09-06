import json
from pathlib import Path

from agent.agent import run_agent
from agent.data import seed_if_empty
from evals.judges import EVAL_CASES, judge_response
from datetime import datetime, timezone


def run_evals():
    seed_if_empty()

    results = []
    passed = 0
    failed = 0
    guardrail_blocks = 0

    print("=" * 60)
    print("Running AgentAudit Evals")
    print("=" * 60)

    for case in EVAL_CASES:
        agent_response = run_agent(case["user_message"])

        blocked = agent_response.startswith("I'm sorry, I'm unable")
        if blocked:
            guardrail_blocks += 1

        result = judge_response(case["user_message"], agent_response, case["expected"])

        status = "PASS" if result["pass"] else "FAIL"
        if result["pass"]:
            passed += 1
        else:
            failed += 1

        case_result = {
            "id": case["id"],
            "user_message": case["user_message"],
            "agent_response": agent_response,
            "expected": case["expected"],
            "passed": result["pass"],
            "reason": result["reason"],
            "guardrail_blocked": blocked,
        }
        results.append(case_result)

        print(f"\n[{status}] {case['id']} — {case['user_message'][:50]}")
        print(f"  Agent: {agent_response[:120]}")
        print(f"  Judge: {result['reason']}")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "guardrail_blocks": guardrail_blocks,
        "cases": results,
    }

    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_evals()
