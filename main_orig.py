# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str     # the actual text

class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "llama-3.1-8b-instant"
    system_prompt: str = "You are an expert AI engineer."

def stream_groq(messages: list[Message], model: str, system_prompt: str):
    full_messages = [
        {"role": "system", "content": system_prompt}
    ] + [
        {"role": m.role, "content": m.content} for m in messages
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=full_messages,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

@app.post("/chat")
def chat(request: ChatRequest):
    return StreamingResponse(
        stream_groq(request.messages, request.model, request.system_prompt),
        media_type="text/plain"
    )

@app.get("/health")
def health():
    return {"status": "ok"}
