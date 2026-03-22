# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
from rag.ingest import ingest_file
from agents.fraud_agent import investigate

from rag.query import retrieve

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── In-Memory Conversation Store ────────────────────────────────────────────
# Lost on server restart — fine for learning
# Production: replace with Redis or PostgreSQL
conversation_store: dict[str, list] = {}


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class BaseRequest(BaseModel):
    session_id: str
    message: str
    temperature: float = 0.7
    system_prompt: str = "You are an expert AI engineer."

class ChatRequest(BaseRequest):
    model: str = "llama-3.1-8b-instant"

class RAGRequest(BaseRequest):
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    system_prompt: str = "You are a fraud investigation expert. Answer using ONLY the context provided. If the answer is not in context say 'I don't have that information.'"

class IngestRequest(BaseModel):
    filepath: str     # e.g. "data/my_document.pdf"

class InvestigateRequest(BaseModel):
    user_id: str
    ip_address: str

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> list:
    """Get existing conversation history or create new session"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── 1. General Chat ───────────────────────────────────────────────────────────

@app.post("/chat")
def chat(request: ChatRequest):
    history = get_or_create_session(request.session_id)

    # Build full messages array with system prompt prepended
    full_messages = [
        {"role": "system", "content": request.system_prompt}
    ] + history + [
        {"role": "user", "content": request.message}   # ← added here directly
    ]

    def stream_and_store():
        full_response = []

        stream = client.chat.completions.create(
            model=request.model,
            messages=full_messages,
            temperature=request.temperature,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                full_response.append(content)
                yield content

        # Save assistant response to history after streaming completes
        history.append({"role": "user",      "content": request.message})
        history.append({
            "role": "assistant",
            "content": "".join(full_response)
        })

    return StreamingResponse(stream_and_store(), media_type="text/plain")


# ── 2. RAG Chat ───────────────────────────────────────────────────────────────

@app.post("/rag/ask")
def rag_ask(request: RAGRequest):
    history = get_or_create_session(request.session_id)
    try:
      # 1. Retrieve relevant chunks from ChromaDB
      relevant_chunks = retrieve(request.message)
    except Exception as e:
      return {"error": "RAG not available — please ingest documents first",
        "hint": "POST /rag/ingest with a filepath"}
    
    if not relevant_chunks:
      return {"error": "No documents ingested yet",
        "hint": "POST /rag/ingest with a filepath"}

    context = "\n\n".join(relevant_chunks)

    # 2. Build messages — system prompt contains retrieved context
    full_messages = [
        {
            "role": "system",
            "content": f"""{request.system_prompt}

Context:
{context}"""
        }
    ] + history + [
        {"role": "user", "content": request.message}
    ]

    def stream_and_store():
        full_response = []

        stream = client.chat.completions.create(
            model=request.model,
            messages=full_messages,
            temperature=request.temperature,
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                full_response.append(content)
                yield content

        # Save to conversation history after streaming completes
        history.append({"role": "user",      "content": request.message})
        history.append({"role": "assistant",  "content": "".join(full_response)})

    return StreamingResponse(stream_and_store(), media_type="text/plain")


# ── 3. Conversation History ───────────────────────────────────────────────────

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """See full conversation history for a session"""
    return conversation_store.get(session_id, [])


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation and start fresh"""
    conversation_store.pop(session_id, None)
    return {"status": "cleared"}

# ── 4. pdf file ingestion ───────────────────────────────────────────────────
@app.post("/rag/ingest")
def rag_ingest(request: IngestRequest):
    response = ingest_file(request.filepath)
    return {"chunks_count": response}

@app.post("/agent/investigate")
def agent_investigate(request: InvestigateRequest):
    response = investigate(request.user_id, request.ip_address)
    return {"result": response}
