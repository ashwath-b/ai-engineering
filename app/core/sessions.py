# app/core/sessions.py

# Lost on server restart — fine for learning
# Production: replace with Redis or PostgreSQL
conversation_store: dict[str, list] = {}

def get_or_create_session(session_id: str) -> list:
    """Get existing conversation history or create new session"""
    if session_id not in conversation_store:
        conversation_store[session_id] = []
    return conversation_store[session_id]
