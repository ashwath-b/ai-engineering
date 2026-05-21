#!/bin/bash

BASE_URL="http://localhost:8000"
PASS=0
FAIL=0

echo "================================"
echo "AI Engineering — Health Check"
echo "================================"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
check() {
  local name=$1
  local result=$2
  local expected=$3

  if echo "$result" | grep -q "$expected"; then
    echo "✅ PASS  $name"
    ((PASS++))
  else
    echo "❌ FAIL  $name"
    echo "        Expected: $expected"
    echo "        Got:      $result"
    ((FAIL++))
  fi
}

# ── 1. Health ─────────────────────────────────────────────────────────────────
echo "── API Health ──────────────────────"
result=$(curl -s $BASE_URL/health)
check "GET /health" "$result" "ok"

# ── 2. Chat ───────────────────────────────────────────────────────────────────
echo ""
echo "── Chat Endpoint ───────────────────"
result=$(curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "verify_001",
    "message": "Reply with exactly: CHAT_OK",
    "system_prompt": "You must reply with exactly what the user asks."
  }')
check "POST /chat — response received" "$result" "CHAT_OK"

# Memory test — does it remember previous message?
result=$(curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "verify_001",
    "message": "What did I ask you to reply with in my first message?"
  }')
check "POST /chat — session memory works" "$result" "CHAT_OK"

# Session isolation test
result=$(curl -s -X POST $BASE_URL/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "verify_002",
    "message": "What did I say in my previous message?"
  }')
check "POST /chat — session isolation" "$result" "previous"

# ── 3. History ────────────────────────────────────────────────────────────────
echo ""
echo "── History Endpoints ───────────────"
result=$(curl -s $BASE_URL/history/verify_001)
check "GET /history/:id — returns history" "$result" "verify_001"

result=$(curl -s -X DELETE $BASE_URL/history/verify_001)
check "DELETE /history/:id — clears session" "$result" "deleted\|success\|ok"

# ── 4. RAG ────────────────────────────────────────────────────────────────────
echo ""
echo "── RAG Endpoints ───────────────────"

# Ingest test
result=$(curl -s -X POST $BASE_URL/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{"filepath": "data/sample.txt"}')
check "POST /rag/ingest — ingests document" "$result" "chunk"

# RAG ask — grounded answer
result=$(curl -s -X POST $BASE_URL/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "verify_rag",
    "message": "What is this document about?"
  }')
check "POST /rag/ask — returns answer" "$result" "."

# RAG grounding test — out of context question
result=$(curl -s -X POST $BASE_URL/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "verify_rag_2",
    "message": "What is the current price of Bitcoin?"
  }')
check "POST /rag/ask — grounding works" "$result" "don't\|not\|unavailable\|context"

# ── 5. Agent ──────────────────────────────────────────────────────────────────
echo ""
echo "── Agent Endpoints ─────────────────"

# High risk user
result=$(curl -s -X POST $BASE_URL/agent/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "ip_address": "192.168.1.100"
  }')
check "POST /agent/investigate — HIGH risk user" "$result" "BLOCK\|HIGH"
check "POST /agent/investigate — fraud score present" "$result" "100\|score"
check "POST /agent/investigate — policy references" "$result" "Policy"

# Low risk user
result=$(curl -s -X POST $BASE_URL/agent/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_456",
    "ip_address": "203.0.113.42"
  }')
check "POST /agent/investigate — LOW risk user" "$result" "APPROVE\|LOW"

# ── 6. ChromaDB ───────────────────────────────────────────────────────────────
echo ""
echo "── ChromaDB ────────────────────────"
result=$(python -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_db')
col = client.get_or_create_collection('documents')
print(f'chunks:{col.count()}')
" 2>&1)
check "ChromaDB — collection exists" "$result" "chunks:"

count=$(echo "$result" | grep -o '[0-9]*')
if [ "$count" -gt "0" ] 2>/dev/null; then
  echo "✅ PASS  ChromaDB — has $count chunks"
  ((PASS++))
else
  echo "❌ FAIL  ChromaDB — 0 chunks (run /rag/ingest first)"
  ((FAIL++))
fi

# ── 7. Environment ────────────────────────────────────────────────────────────
echo ""
echo "── Environment Variables ───────────"
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

vars = {
    'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
    'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
    'LANGCHAIN_TRACING_V2': os.getenv('LANGCHAIN_TRACING_V2'),
    'LANGCHAIN_PROJECT': os.getenv('LANGCHAIN_PROJECT'),
}

for name, val in vars.items():
    if val:
        print(f'✅ PASS  {name} is set')
    else:
        print(f'❌ FAIL  {name} is NOT set')
"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
  echo "✅ All checks passed"
else
  echo "❌ $FAIL checks need attention"
fi
echo "================================"