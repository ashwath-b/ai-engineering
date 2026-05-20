# evals/run_harness.py
import asyncio
import json
from evals.judge_prompt import JUDGE_SYSTEM_PROMPT
from app.agent import graph # Your existing LangGraph

# 1. Define your Golden Set
GOLDEN_SET = [
    {
        "input": "Check user_123 for fraud on a $5000 tx.",
        "expected_verdict": "HOLD",
        "required_tools": ["get_user_history", "calculate_fraud_score"]
    }
]

async def run_eval():
    results = []
    for case in GOLDEN_SET:
        # Run actual agent
        response = await graph.ainvoke({"messages": [("user", case["input"])]})
        
        # Prepare data for Judge
        actual_output = response["messages"][-1].content
        actual_tools = [m.tool_calls[0]['name'] for m in response['messages'] if hasattr(m, 'tool_calls')]

        # Call Judge (Standard OpenAI/Anthropic call here)
        # judge_result = await call_judge(actual_output, actual_tools, case)
        
        print(f"✅ Tested: {case['input']} | Score: {judge_result['score']}")
        results.append(judge_result)

    # Final summary
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"--- FINAL EVAL SCORE: {avg_score * 100}% ---")

if __name__ == "__main__":
    asyncio.run(run_eval())