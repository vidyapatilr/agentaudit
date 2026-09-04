from agent.agent import run_agent
from agent.data import seed_if_empty
from evals.judges import EVAL_CASES, judge_response


def run_evals():
    seed_if_empty()

    passed = 0
    failed = 0

    print("=" * 60)
    print("Running AgentAudit Evals")
    print("=" * 60)

    for case in EVAL_CASES:
        agent_response = run_agent(case["user_message"])
        result = judge_response(case["user_message"], agent_response, case["expected"])

        status = "PASS" if result["pass"] else "FAIL"
        if result["pass"]:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {case['id']} — {case['user_message'][:50]}")
        print(f"  Agent: {agent_response[:120]}")
        print(f"  Judge: {result['reason']}\n")

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("=" * 60)


if __name__ == "__main__":
    run_evals()
