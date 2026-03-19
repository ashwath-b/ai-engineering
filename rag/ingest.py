# rag/ingest.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
from pypdf import PdfReader

# Load embedding model (downloads once, ~90MB, runs locally — free)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Local ChromaDB — stores data in ./chroma_db folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

def chunk_text(text: str, chunk_size: int = 50) -> list[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    overlap = 20  # words shared between chunks for context continuity

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

def ingest_file(filepath: str) -> int:
    """Load a file, chunk it, embed it, store in ChromaDB"""

    # 1. Load
    if filepath.lower().endswith('.pdf'):
        text = load_pdf(filepath)
    else:
        with open(filepath, "r") as f:
            text = f.read()

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunks")

    # 3. Embed + 4. Store
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.upsert(
            ids=[f"{filepath}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filepath, "chunk_index": i}]
        )

    print(f"Stored {len(chunks)} chunks from {filepath}")
    return len(chunks)

def load_pdf(filepath: str) -> str:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

if __name__ == "__main__":
    ingest_file("data/sample.txt")