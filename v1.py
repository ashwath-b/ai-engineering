# test_setup.py
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say 'Setup successful!' and nothing else."}
    ]
)

print(message.content[0].text)
