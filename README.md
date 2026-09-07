# AgentAudit

A production-inspired eval and guardrail harness for a custoer support LLM agent.

Built to demonstrate how to move beyond "doesit seem to work?" and into measurable, enforceable AI reliability.

---

## What it does

AgentAudit wraps a customer support agent with four layers of production engineering:

- **Tracing** - every LLM call and tool execution is captured in Arize Phoenix
- **Guradrails** - rule-based and LLM-judge checks block policy-violating responses before the customer sees them
- **Evals** - an 11-case golden dataset scores the agent on every run
- **CI** - Github Actions run the full eval suite on every push and commits results back to the repo

---

## Architecture

User message
↓
Agent (gpt-4o-mini via OpenRouter)
↓
Tool calls → PostgreSQL (order lookup, refund eligibility)
↓
Guardrail Layer 1: rule-based checks (PII, forbidden phrases, high-value refunds)
↓
Guardrail Layer 2: LLM-judge policy classifier
↓
Response returned to user
↓
Arize Phoenix (tracing) + Streamlit (eval dashboard)

---

# Stack

| Component       |            Technology             |
|-----------------|-----------------------------------|
| Agent           | gpt-4o-mini via OpenRouter        |
| Database        | PostgreSQL 16 (Docker)            |
| Tracing         | Arize Phoenix (local)             |
| Guardrails      | Rule-based + LLM-judge            |
| Evals           | Custom golden dataset + LLM-judge |
| Dashboard       | Streamlit                         |
| CI              | GitHub Actions.                   |
| Package manager | uv                                |

---

## How to run locally

**Prerequisites:** Docker, Python 3.12, uv, an OpenRouter API key

**1. Start the database**
```bash
docker compose up -d

**2. Start Tracing**
```bash
uv run phoenix serve

**3. Run the agent**
```bash
uv run python -m agent.agent

**4. Run Evals**
```bash
uv run python -m evals.run_evals

**5. View the dashboard**
```bash
uv run streamlit run dashboard/app.py

---

## Project Structure
agent/          # LLM agent, tools, data layer, policy
evals/          # Golden dataset, LLM judge, eval runner, results
guradrails/     # Rule-based checks and LLM classifier
tracing/        # Arize Phoenix setup
dashboard/      # Streamlit eval dashboard
.github/        # CI workflow

## CI
Every push to main triggers a full eval run on GitHub Actions. Results are committed back to evals/results.json and reflected in the dashboard.

## What I would do differently in production
Use Arize cloud instead of local Phoenix so CI traces are captured
Add structured logging for operational errors (database down, API timeouts)
Expand the golden dataset with more adversarial and edge cases
