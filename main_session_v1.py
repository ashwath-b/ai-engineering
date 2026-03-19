from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory store — lost on server restart, fine for learning
conversation_store: dict[str, list] = {}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str                        # identifies the conversation
    message: str                           # just the new user message
    model: str = "llama-3.1-8b-instant"
    system_prompt: str = "You are an expert AI engineer."

def stream_groq(messages: list, model: str, system_prompt: str):
    full_messages = [
        {"role": "system", "content": system_prompt}
    ] + messages

    stream = client.chat.completions.create(
        model=model,
        messages=full_messages,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content

@app.post("/chat")
def chat(request: ChatRequest):
    # 1. Get or create conversation history for this session
    if request.session_id not in conversation_store:
        conversation_store[request.session_id] = []

    # 2. Append the new user message to history
    conversation_store[request.session_id].append({
        "role": "user",
        "content": request.message
    })

    # 3. Collect full response so we can store it
    full_response = []

    def stream_and_store():
        for chunk in stream_groq(
            conversation_store[request.session_id],
            request.model,
            request.system_prompt
        ):
            full_response.append(chunk)
            yield chunk

        # 4. After streaming done, save assistant response to history
        conversation_store[request.session_id].append({
            "role": "assistant",
            "content": "".join(full_response)
        })

    return StreamingResponse(
        stream_and_store(),
        media_type="text/plain"
    )

@app.get("/history/{session_id}")
def get_history(session_id: str):
    # Bonus endpoint — see full conversation history
    return conversation_store.get(session_id, [])

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    # Clear a conversation and start fresh
    conversation_store.pop(session_id, None)
    return {"status": "cleared"}

@app.get("/health")
def health():
    return {"status": "ok"}
