"""
Ingest documents from sample_docs/ into a local Chroma vector store.
Supports .txt, .md, and .pdf files.

Run this once (or whenever documents change):
    python src/ingest.py
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pypdf import PdfReader

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

# Chunking config
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple sliding-window chunker. Good enough for a portfolio project;
    swap for a semantic/recursive chunker if you want to go further."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents():
    txt_paths = glob.glob(os.path.join(DOCS_DIR, "*.txt")) + glob.glob(os.path.join(DOCS_DIR, "*.md"))
    pdf_paths = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    paths = txt_paths + pdf_paths
    if not paths:
        raise FileNotFoundError(
            f"No .txt, .md, or .pdf files found in {DOCS_DIR}. Add some documents first."
        )
    docs = []
    for path in paths:
        if path.lower().endswith(".pdf"):
            text = extract_pdf_text(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def main():
    print("Loading embedding model (all-MiniLM-L6-v2, CPU)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading documents...")
    docs = load_documents()

    print("Chunking...")
    all_chunks, all_metadatas, all_ids = [], [], []
    chunk_id = 0
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})
            all_ids.append(f"chunk_{chunk_id}")
            chunk_id += 1

    print(f"Total chunks created: {len(all_chunks)}")

    print("Embedding chunks...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True).tolist()

    print("Writing to Chroma vector store...")
    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh collection each time ingest is run
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    for i in tqdm(range(0, len(all_chunks), 100)):
        collection.add(
            ids=all_ids[i:i + 100],
            documents=all_chunks[i:i + 100],
            embeddings=embeddings[i:i + 100],
            metadatas=all_metadatas[i:i + 100],
        )

    print(f"Done. Ingested {len(all_chunks)} chunks from {len(docs)} document(s) into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()