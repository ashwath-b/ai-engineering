---
name: fraud-investigation-agent
description: "Use this skill when asked to investigate
a user for fraud, analyze transaction patterns, check IP
reputation, or produce a fraud risk assessment. Triggers:
'investigate user', 'fraud check', 'risk assessment',
'is this user fraudulent', 'analyze transactions'."
---

# Fraud Investigation Agent Skill

## Overview

This skill enables autonomous fraud investigation using
a LangGraph ReAct agent with 6 tools. The agent collects
transaction data, IP reputation, account info, and policy
references — then produces a structured risk report.

## When to Use This Skill

Use when:
- Given a user_id and IP address to investigate
- Asked to assess fraud risk for a specific user
- Asked to explain why a user was blocked or approved
- Asked to check if transaction patterns are suspicious

Do NOT use when:
- The question is about fraud in general (use RAG instead)
- No specific user_id is provided
- Asked about model training or ML algorithms

## API Endpoint

The agent is exposed via FastAPI at:
```
POST /agent/investigate
Content-Type: application/json

{
  "user_id": "user_123",
  "ip_address": "192.168.1.100"
}
```

Live URL:
```
https://ai-engineering-production.up.railway.app/agent/investigate
```

## Tool Execution Order — Critical

The agent MUST follow this exact order:
```
Step 1 — Parallel data gathering:
  get_transaction_history(user_id)
  check_ip_reputation(ip_address)
  get_account_info(user_id)

Step 2 — After receiving real data:
  calculate_fraud_score(
    return_rate=ACTUAL value from step 1,
    account_age_days=ACTUAL value from step 1,
    non_receipted_rate=ACTUAL value from step 1,
    unique_stores=ACTUAL value from step 1
  )

Step 3 — Policy check:
  search_fraud_policy(query=specific signals found)

Step 4 — Write report
```

NEVER call calculate_fraud_score before step 1 returns.
Using default/assumed values produces wrong scores.

## Decision Rules

Apply these strictly:
```
fraud_score >= 80          → BLOCK
fraud_score 50-79          → REVIEW
fraud_score < 50           → APPROVE

Override to BLOCK if:
  account_age_days < 7 AND return_rate > 0.5

Override to REVIEW if:
  vpn_detected = true AND any other signal present
```

## Output Format — Always Use This Exactly
```
User ID: {user_id}
Login IP: {ip_address}
Fraud Score: {0-100}
Risk Level: HIGH / MEDIUM / LOW
Key Signals Found:
- {signal 1 with specific numbers}
- {signal 2 with specific numbers}
Policy References:
- {relevant policy text}
Recommended Action: BLOCK / REVIEW / APPROVE
Confidence: {0-100}%
Brief Justification: {2-3 sentences max}
```

## Known Gotchas

**Gotcha 1 — Parallel tool calls with dependent tools**
The LLM may call calculate_fraud_score in parallel
with data tools before it has real values. This produces
fraud_score=0 even for clear fraudsters. Fix: explicit
ordering instruction in system prompt.

**Gotcha 2 — recursion_limit**
Minimum graph nodes needed = 4.
Production setting = 10 (2.5x safety margin).
Setting below 4 causes GraphRecursionError immediately.

**Gotcha 3 — String return types**
All tools return json.dumps() strings, not Python dicts.
LLMs work with text. Never return raw Python objects.

## Adding New Tools

When adding a new tool to the agent:

1. Write docstring first — it IS the tool's interface
2. Return json.dumps() always
3. Wrap in try/except — never let a tool crash the agent
4. Add to tools list in fraud_agent.py
5. Update system_prompt ordering if tool depends on others
6. Run investigation and check LangSmith trace
7. Verify tool was called with correct arguments

Template:
```python
@tool
def your_new_tool(param: str) -> str:
    """Clear description of what this tool does.
    When to call it. What it returns. Dependencies."""
    try:
        result = your_logic_here(param)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e), "param": param})
```

## Extending the RAG Pipeline

To add new documents to the knowledge base:
```bash
# Via API
curl -X POST https://ai-engineering-production.up.railway.app/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/your_document.pdf"}'

# Locally
python rag/ingest.py  # add filepath to __main__ block
```

Key parameters to tune per document type:
```
Short articles:    chunk_size=500,  overlap=50
Research papers:   chunk_size=2000, overlap=200
Policy documents:  chunk_size=1500, overlap=150
Code files:        chunk_size=500,  overlap=100
```

## RAG Query Tips

For best retrieval results:
- Be specific in queries: "84% return rate fraud" not "high returns"
- Include numbers when known: improves semantic match
- Use search_fraud_policy for raw chunks (fast, cheap)
- Use ask_fraud_policy for synthesized answers (slower, costs tokens)

## Environment Setup

Required environment variables:
```
GROQ_API_KEY              # Groq/Llama inference
LANGCHAIN_TRACING_V2=true # LangSmith tracing
LANGCHAIN_API_KEY         # LangSmith API key
LANGCHAIN_PROJECT         # Project name in LangSmith
ANTHROPIC_API_KEY         # Optional, for Claude models
```

## LangSmith Debugging

When agent produces unexpected output:

1. Go to smith.langchain.com → your project
2. Find the run by run_name (fraud_investigation_{user_id})
3. Check LLM Call 2 → what args did calculate_fraud_score receive?
4. If return_rate=0.05 instead of 0.84 → ordering bug
5. Fix: strengthen the ordering instruction in system_prompt
6. Re-run and verify in next trace

Key metrics to check per run:
```
LLM calls:     should be 3 (data → score+policy → report)
Total tokens:  3000-5000 with RAG
fraud_score:   100 for user_123, 0 for user_456
Confidence:    100% when all signals align
```