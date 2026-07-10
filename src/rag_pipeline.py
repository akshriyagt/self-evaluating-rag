"""
Core pipeline: embed query -> retrieve chunks -> generate answer ->
judge grounding -> return structured result.
"""

import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from .judge import JUDGE_SYSTEM_PROMPT, build_judge_prompt, parse_judge_response

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

# Ollama runs locally -- no API key, no cost. Make sure you've pulled a model
# first, e.g.:  ollama pull llama3
_raw_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
if not _raw_host.startswith("http://") and not _raw_host.startswith("https://"):
    _raw_host = f"http://{_raw_host}"
OLLAMA_HOST = _raw_host
GENERATION_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
JUDGE_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
TOP_K = 3
CONFIDENCE_THRESHOLD = 60  # below this, answer is flagged as uncertain

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions \
based ONLY on the provided context. If the context does not contain the \
answer, say clearly that you don't have enough information to answer -- do \
not guess or use outside knowledge."""


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)

    def _ollama_chat(self, model: str, system: str, user_content: str) -> str:
        """Call a local Ollama model via its REST API. No API key needed."""
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def retrieve(self, query: str, top_k: int = TOP_K):
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        chunks = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        return chunks, metadatas

    def generate_answer(self, query: str, context: str, chat_history: str = "") -> str:
        history_block = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n" if chat_history else ""
        return self._ollama_chat(
            model=GENERATION_MODEL,
            system=ANSWER_SYSTEM_PROMPT,
            user_content=f"CONTEXT:\n{context}\n{history_block}\nQUESTION:\n{query}",
        )

    def judge_answer(self, query: str, context: str, answer: str) -> dict:
        prompt = build_judge_prompt(query, context, answer)
        raw_text = self._ollama_chat(
            model=JUDGE_MODEL,
            system=JUDGE_SYSTEM_PROMPT,
            user_content=prompt,
        )
        return parse_judge_response(raw_text)

    def answer_query(self, query: str, chat_history: str = "") -> dict:
        chunks, metadatas = self.retrieve(query)
        context = "\n\n---\n\n".join(chunks) if chunks else ""

        if not context:
            return {
                "query": query,
                "answer": "No relevant documents found in the vector store.",
                "grounded": False,
                "confidence": 0,
                "reasoning": "No chunks retrieved.",
                "flagged_uncertain": True,
                "sources": [],
            }

        answer = self.generate_answer(query, context, chat_history)
        judgment = self.judge_answer(query, context, answer)

        flagged = judgment["confidence"] < CONFIDENCE_THRESHOLD or not judgment["grounded"]

        return {
            "query": query,
            "answer": answer,
            "grounded": judgment["grounded"],
            "confidence": judgment["confidence"],
            "reasoning": judgment["reasoning"],
            "flagged_uncertain": flagged,
            "sources": [m.get("source", "unknown") for m in metadatas],
            "retrieved_chunks": chunks,
        }