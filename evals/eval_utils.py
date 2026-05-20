# Create a new file: evals/eval_utils.py

def extract_logic_path(state):
    """Extracts tool names used in the conversation."""
    messages = state.get("messages", [])
    tool_calls = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append(tc['name'])
    return tool_calls