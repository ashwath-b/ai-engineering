# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq, BadRequestError
from dotenv import load_dotenv
import os

load_dotenv()
app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
  role: str
  content: str

class ChatRequest(BaseModel):
  messages: list[Message]
  model: str = "llama-3.1-8b-instant"
  system_prompt: str = "You are a senior AI Engineer"

def with_groq(messages, model: str, system_prompt: str):
  try:
    full_message = [
        {"role": "system", "content": system_prompt}
    ] + [
        {"role": m.role, "content": m.content} for m in messages
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=full_message,
        stream=True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

  except BadRequestError as e:
      yield f"\n[Groq Error] {e}"
  
@app.post("/chat")
def chat(request: ChatRequest):
  return StreamingResponse(
    with_groq(request.messages, request.model, request.system_prompt),
    media_type="text/plain"
  )