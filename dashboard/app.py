import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="AgentAudit Dashboard", layout="wide")

st.title("AgentAudit Dashboard")
st.caption("Eval results and guardrail summary for the customer support agent")
st.caption(f"Last run: {data.get('run_at', 'unknown')}")

results_path = Path(__file__).resolve().parent.parent / "evals" / "results.json"
print(f"Looking for results at: {results_path}")


if not results_path.exists():
    st.warning("No results found. Run `uv run python -m evals.run_evals` first.")
    st.stop()

data = json.loads(results_path.read_text())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cases", data["total"])
col2.metric("Passed", data["passed"])
col3.metric("Failed", data["failed"])
col4.metric("Guardrail Blocks", data["guardrail_blocks"])

st.divider()

st.subheader("Eval Cases")

for case in data["cases"]:
    status = "PASS" if case["passed"] else "FAIL"
    color = "green" if case["passed"] else "red"
    with st.expander(f"[{status}] {case['id']} — {case['user_message']}"):
        st.markdown(f"**Status:** :{color}[{status}]")
        if case["guardrail_blocked"]:
            st.warning("Guardrail blocked the original response")
        st.markdown(f"**Agent response:** {case['agent_response']}")
        st.markdown(f"**Expected:** {case['expected']}")
        st.markdown(f"**Judge reasoning:** {case['reason']}")
