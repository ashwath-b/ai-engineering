# rag/langchain_rag.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ── Setup ──────────────────────────────────────────────────────────────────────

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db_langchain"
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Prompt ─────────────────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_template("""
You are a fraud investigation expert.
Answer using ONLY the context below.
If the answer is not in context say 'I don't have that information.'

Context: {context}
Question: {question}
""")

# ── Helpers ────────────────────────────────────────────────────────────────────

def format_docs(docs):
    """Convert list of Document objects to a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# ── Ingest ─────────────────────────────────────────────────────────────────────

def ingest(filepath: str) -> int:
    # 1. Load
    if filepath.lower().endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath)

    documents = loader.load()

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(documents)

    ids = [
        f"{filepath}_chunk_{i}"
        for i in range(len(chunks))
    ]

    # 3. Embed + Store
    vectorstore.add_documents(chunks, ids=ids)

    print(f"Stored {len(chunks)} chunks from {filepath}")
    return len(chunks)

# ── Query ──────────────────────────────────────────────────────────────────────

def ask(question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )

    # ── DEBUG — print what's actually being retrieved ──
    retrieved_docs = retriever.invoke(question)
    print("\n=== RETRIEVED CHUNKS ===")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content[:500])  # first 200 chars
    print("========================\n")
    # ── END DEBUG ──

    # LCEL chain — read left to right like a pipeline
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

# ── Test ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Only ingest if --ingest flag passed
    # python rag/langchain_rag.py --ingest
    # python rag/langchain_rag.py          ← just query, no ingest
    if "--ingest" in sys.argv:
        ingest("data/fraud_detection.pdf")
        print("Ingestion complete.")
    else:
        questions = [
            "What variables are most predictive of fraud?",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            print(f"A: {ask(q)}")
            print("-" * 50)