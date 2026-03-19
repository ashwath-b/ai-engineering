# AI Engineering Portfolio

A production-grade AI backend built during a hands-on transition from
Ruby on Rails backend engineering to AI engineering.

Built from scratch in 2 weeks — no tutorials, no boilerplate.
Every component understood and written manually first,
then rebuilt with production frameworks.

---

## What This Is

A FastAPI backend that demonstrates three core AI engineering patterns:

1. **Streaming Chat** — conversational AI with persistent memory
2. **RAG Pipeline** — document-grounded Q&A that doesn't hallucinate
3. **Autonomous Agent** — fraud investigation using tool calling

---

## Architecture
```
Client
  │
  ▼
FastAPI (main.py)
  ├── /chat              → Groq LLM + conversation memory
  ├── /rag/ask           → ChromaDB retrieval + Groq LLM
  ├── /rag/ingest        → PDF/TXT → chunks → embeddings → ChromaDB
  └── /agent/investigate → LangGraph agent → 4 tools → fraud report
  
RAG Stack
  PDF → PyPDF → RecursiveTextSplitter → HuggingFace Embeddings
      → ChromaDB → semantic retrieval → LLM synthesis

Agent Stack
  Task → LangGraph ReAct loop → tool selection → tool execution
       → result synthesis → structured report
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| LLM Provider | Groq (Llama 3.3 70B) |
| Agent Framework | LangChain + LangGraph |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF Processing | PyPDF |
| Validation | Pydantic v2 |
| Package Manager | uv |

---

## Project Structure
```
ai-engineering/
├── main.py              # FastAPI app — all endpoints
├── agents/
│   └── fraud_agent.py   # LangGraph fraud investigation agent
├── rag/
│   ├── ingest.py        # Document ingestion pipeline
│   ├── query.py         # Retrieval + answer generation
│   └── langchain_rag.py # LangChain LCEL RAG implementation
├── data/
│   └── sample.txt       # Sample fraud report for testing
├── requirements.txt
└── .gitignore
```

---

## Setup

**Prerequisites:** Python 3.11+, uv
```bash
# Clone
git clone https://github.com/ashwath-b/ai-engineering.git
cd ai-engineering

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys:
# GROQ_API_KEY=gsk_...
# ANTHROPIC_API_KEY=sk-ant-... (optional)

# Ingest sample document
python rag/ingest.py

# Start server
uvicorn main:app --reload
```

---

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

---

### 1. Streaming Chat with Memory

Stateful conversation with session isolation.
Each `session_id` maintains independent conversation history.
```bash
# Turn 1
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_001",
    "message": "My name is Arjun. I am learning AI engineering.",
    "system_prompt": "You are a helpful AI engineering mentor."
  }' --no-buffer

# Turn 2 — model remembers context
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_001",
    "message": "What am I learning?"
  }' --no-buffer
```

---

### 2. RAG — Ingest a Document
```bash
curl -X POST http://localhost:8000/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/sample.txt"}'
```

Response:
```json
{"chunks_count": 7, "filepath": "data/sample.txt"}
```

---

### 3. RAG — Ask Questions Grounded in Your Documents
```bash
# Answers from your document only — won't hallucinate
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "analyst_001",
    "message": "What fraud patterns increased in Q3 2024?"
  }' --no-buffer

# Out-of-context question — model correctly says it doesn't know
curl -X POST http://localhost:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "analyst_001",
    "message": "What is the stock price of Apple?"
  }' --no-buffer
```

---

### 4. Autonomous Fraud Investigation Agent

Agent autonomously calls tools, investigates, and produces
a structured fraud report without hardcoded steps.
```bash
# High risk user
curl -X POST http://localhost:8000/agent/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "ip_address": "192.168.1.100"
  }'

# Low risk user
curl -X POST http://localhost:8000/agent/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_456",
    "ip_address": "203.0.113.42"
  }'
```

Example response:
```json
{
  "status": "success",
  "result": "User ID: user_123\nRisk Level: HIGH\nKey Fraud Signals Found:\n- High return rate of 84%\n- Non-receipted return rate of 71%\n- High-risk IP with VPN detected\n- Account age: 3 days, unverified email/phone\n- Multiple failed logins\nRecommended Action: BLOCK\nConfidence: 95%"
}
```

---

### 5. Conversation History
```bash
# View history
curl http://localhost:8000/history/user_001

# Clear history
curl -X DELETE http://localhost:8000/history/user_001
```

---

## Key Engineering Decisions

**Why Groq over OpenAI?**
Groq's LPU hardware runs open-source models (Llama 3.3) at
10-20x the speed of GPU-based providers. Free tier is generous
for development. Zero system resource usage vs running models locally.

**Why ChromaDB over Pinecone?**
Local persistence with zero setup for development.
Same retrieval API as production vector stores —
swap to Pinecone/Weaviate in one line for production.

**Why sentence-transformers for embeddings?**
Runs locally, zero API cost, zero latency.
`all-MiniLM-L6-v2` is fast and accurate enough for most RAG use cases.

**Why session-based memory over persistent DB?**
In-memory is correct for learning and development.
The session pattern is identical to production —
only the storage backend changes (Redis/PostgreSQL).

---

## What I Learned Building This

Coming from 8 years of Ruby on Rails, the core insight was:

> AI engineering is systems engineering with probabilistic
> components. Every pattern I knew — async queues, caching,
> API design, observability — applies directly. The new skill
> is understanding where the LLM fits in the pipeline and how
> to constrain its behavior reliably.

Key concepts mastered:
- RAG pipeline design and retrieval tuning
- Agent ReAct loop and tool calling patterns
- Prompt engineering for consistent structured output
- Chunking strategy tradeoffs for different document types
- LangChain LCEL pipeline composition
- LangGraph agent graph and recursion control

---

## Roadmap
```
✅ Week 1-2  Core API + RAG + Agent
⬜ Week 3    LangSmith tracing + error handling
⬜ Week 4    Connect RAG to Agent (policy-aware investigations)
⬜ Week 5    Docker + AWS deployment
⬜ Week 6    Evaluation pipeline + regression tests
```

---

## Background

This project was built as part of a deliberate transition from
backend engineering (Ruby on Rails, 8 years) to AI engineering.

Target domain: Trust & Safety, Fraud Detection, AI Agentic Systems.

Built by following a structured learning path:
environment setup → LLM APIs → streaming → RAG from scratch →
LangChain abstractions → LangGraph agents → production patterns.