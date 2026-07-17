# main.py

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from app.schemas import BaseRequest, ChatRequest, RAGRequest, IngestRequest, InvestigateRequest, Chunk, Message


from agents.fraud_agent import investigate
from rag.ingest import ingest_file
from rag.query import retrieve, retrieve_async

load_dotenv()

# Constants
DEFAULT_CHAT_MODEL = "llama-3.1-8b-instant"
DEFAULT_RAG_MODEL = "llama-3.3-70b-versatile"

app = FastAPI()
client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

from app.api.chat import router as chat_router

app.include_router(chat_router)

from app.core.sessions import conversation_store, get_or_create_session

@app.get("/health")
def health():
    return {"status": "ok"}

# ── 2. RAG Chat ───────────────────────────────────────────────────────────────

@app.post("/rag/ask")
async def rag_ask(request: RAGRequest):
    history = get_or_create_session(request.session_id)
    try:
      # 1. Retrieve relevant chunks from ChromaDB
      relevant_chunks = await retrieve_async(request.message)
    except Exception as e:
      return {"error": "RAG not available — please ingest documents first",
        "hint": "POST /rag/ingest with a filepath"}

    if not relevant_chunks:
      return {"error": "No documents ingested yet",
        "hint": "POST /rag/ingest with a filepath"}

    context = "\n\n".join(c.text for c in relevant_chunks)

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

    async def stream_and_store():
        full_response = []

        stream = await client.chat.completions.create(
            model=request.model,
            messages=full_messages,
            temperature=request.temperature,
            stream=True
        )

        async for chunk in stream:
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
    return {"session_id": session_id, "history": conversation_store.get(session_id, [])}


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation and start fresh"""
    conversation_store.pop(session_id, None)
    return {"status": "deleted"}

# ── 4. pdf file ingestion ───────────────────────────────────────────────────
@app.post("/rag/ingest")
def rag_ingest(request: IngestRequest):
    response = ingest_file(request.filepath)
    return {"chunks_count": response}

@app.post("/agent/investigate")
def agent_investigate(request: InvestigateRequest):
    response = investigate(request.user_id, request.ip_address)
    return {"result": response}
