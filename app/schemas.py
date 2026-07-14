from pydantic import BaseModel, Field
from app.core.config import DEFAULT_CHAT_MODEL, DEFAULT_RAG_MODEL

class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class BaseRequest(BaseModel):
    session_id: str
    message: str
    temperature: float = 0.7
    system_prompt: str = "You are an expert AI engineer."

class ChatRequest(BaseRequest):
    model: str = DEFAULT_CHAT_MODEL

class RAGRequest(BaseRequest):
    model: str = DEFAULT_RAG_MODEL
    temperature: float = 0.3
    system_prompt: str = "You are a fraud investigation expert. Answer using ONLY the context provided. If the answer is not in context say 'I don't have that information.'"

class IngestRequest(BaseModel):
    filepath: str     # e.g. "data/my_document.pdf"

class InvestigateRequest(BaseModel):
    user_id: str
    ip_address: str

class Chunk(BaseModel):
  text: str
  score: float = Field(ge=0.0, le=1.0)
  source: str
