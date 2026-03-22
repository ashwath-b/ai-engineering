# rag/query.py
import chromadb
from fastembed import TextEmbedding
from fastembed import TextEmbedding
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def retrieve(question: str, n_results: int = 5) -> list[str]:
    """Find most relevant chunks for a question"""

    # Embed the question using same model as ingestion
    question_embedding = list(embedding_model.embed([question]))[0].tolist()

    # Find closest chunks in vector space
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    return results["documents"][0]  # list of relevant chunks

def ask(question: str) -> str:
    """Retrieve relevant context and answer the question"""

    # 1. Retrieve relevant chunks
    relevant_chunks = retrieve(question)
    context = "\n\n".join(relevant_chunks)

    print(f"\n--- Retrieved Context ---\n{context}\n------------------------\n")

    # 2. Build prompt with context injected
    messages = [
        {
            "role": "system",
            "content": """You are a fraud investigation expert.
Answer questions using ONLY the context provided.
If the answer is not in the context, say 'I don't have that information.'
Always cite which part of the context supports your answer."""
        },
        {
            "role": "user",
            "content": f"""Context:
{context}

Question: {question}"""
        }
    ]

    # 3. Generate answer
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # smarter model for better answers
        messages=messages
    )

    return response.choices[0].message.content

# if __name__ == "__main__":
#     # Test it
#     questions = [
#         "What fraud patterns increased in Q3 2024?",
#         "How do I detect synthetic identity fraud?",
#         "What transactions should I flag as high risk?",
#         "What is the weather like today?",  # should say "I don't have that information"
#     ]

#     for q in questions:
#         print(f"\nQ: {q}")
#         print(f"A: {ask(q)}")
#         print("-" * 50)

def ask_with_temperature(question: str, temperature: float) -> str:
    relevant_chunks = retrieve(question)
    context = "\n\n".join(relevant_chunks)

    messages = [
        {"role": "system", "content": "You are a fraud expert. Be concise, one sentence only."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    question = "What fraud patterns increased in Q3 2024?"

    print("=== temperature=0.0 (run 3 times) ===")
    for _ in range(3):
        print(ask_with_temperature(question, temperature=0.0))

    print("\n=== temperature=1.0 (run 3 times) ===")
    for _ in range(3):
        print(ask_with_temperature(question, temperature=1.0))