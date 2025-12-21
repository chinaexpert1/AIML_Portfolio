"""app.py – unified RAG service (Mistral edition)
--------------------------------------------------
Flask application that exposes a `/generate` endpoint implementing the
full retrieval‑augmented generation pipeline **powered by Mistral’s chat
API**. The endpoint now supports **multiple reranking strategies**
(`cross_encoder`, `tfidf`, `bow`, `hybrid`, or `sequential`).

Send JSON like:
```
POST /generate
{
  "query": "What is the role of antioxidants in green tea?",
  "rerank_type": "hybrid",        // optional, default "sequential"
  "seq_k1": 15,                   // only used for sequential
  "seq_k2": 5                     // only used for sequential
}
```
"""

from __future__ import annotations

import os
os.environ["MISTRAL_API_KEY"] = "GDTId8eQPtNGoVAhqkr5hel3mKqtoD1j"
import glob
from typing import List
import mistralai
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

import numpy as np
import requests

# third‑party / local modules
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import FaissBruteForce
from reranker import Reranker  # implemented previously

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STORAGE_DIRECTORY = "storage"
CHUNKING_STRATEGY = "fixed-length"  # "fixed-length" | "sentence"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 20
SEQ_K1_DEFAULT = 10
SEQ_K2_DEFAULT = 5

# Mistral settings
MISTRAL_MODEL = "mistral-medium"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Chunking params
FIXED_CHUNK_SIZE = 500
FIXED_OVERLAP = 2
SENTENCE_NUM = 5
SENTENCE_OVERLAP = 3

# Allowed rerankers
ALLOWED_RERANKERS = {"cross_encoder", "tfidf", "bow", "hybrid", "sequential"}

# ---------------------------------------------------------------------------
# Flask app & globals
# ---------------------------------------------------------------------------
app = Flask(__name__)

_index: FaissBruteForce | None = None
_chunks: List[str] = []
_embedder: Embedding | None = None

# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def initialize_index():
    global _index, _chunks, _embedder
    if _index is not None:
        return _index

    corpus_dir = os.path.join(STORAGE_DIRECTORY, "corpus")
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory '{corpus_dir}' not found.")

    processor = DocumentProcessing()
    _embedder = Embedding(EMBEDDING_MODEL)

    embeddings: List[np.ndarray] = []
    metadata: List[str] = []

    for file_path in glob.glob(os.path.join(corpus_dir, "*")):
        if not file_path.lower().endswith(".txt"):
            continue
        if CHUNKING_STRATEGY == "sentence":
            chunks = processor.sentence_chunking(file_path, num_sentences=SENTENCE_NUM, overlap_size=SENTENCE_OVERLAP)
        else:
            chunks = processor.fixed_length_chunking(file_path, chunk_size=FIXED_CHUNK_SIZE, overlap_size=FIXED_OVERLAP)

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            embeddings.append(_embedder.encode(chunk))
            metadata.append(chunk)

    if not embeddings:
        raise RuntimeError("No embeddings generated from corpus.")

    dim = embeddings[0].shape[0]
    _index = FaissBruteForce(dim, metric="euclidean")
    _index.add_embeddings(embeddings, metadata)
    _chunks = metadata
    return _index

# ---------------------------------------------------------------------------
# Helper – call Mistral
# ---------------------------------------------------------------------------

def call_mistral(messages: list[dict]) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY env var not set.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MISTRAL_MODEL, "messages": messages}
    resp = requests.post(MISTRAL_API_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Mistral API error {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# /generate endpoint
# ---------------------------------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate_answer():
    if not request.is_json:
        raise BadRequest("Content‑Type must be application/json")
    payload = request.get_json(silent=True) or {}

    query = str(payload.get("query", "")).strip()
    if not query:
        raise BadRequest("JSON body must contain a non‑empty 'query' field.")

    rerank_type = str(payload.get("rerank_type", "sequential")).lower()
    if rerank_type not in ALLOWED_RERANKERS:
        raise BadRequest(f"rerank_type must be one of {sorted(ALLOWED_RERANKERS)}")

    seq_k1 = int(payload.get("seq_k1", SEQ_K1_DEFAULT))
    seq_k2 = int(payload.get("seq_k2", SEQ_K2_DEFAULT))

    # Ensure index
    index = initialize_index()
    embedder = _embedder  # type: ignore

    # Retrieve candidates
    q_vec = embedder.encode(query)
    _, ids = index.search(q_vec, TOP_K_RETRIEVAL)
    candidate_chunks = [_chunks[i] for i in ids]

    # Rerank
    reranker = Reranker(type=rerank_type)
    if rerank_type == "sequential":
        ranked_docs, _, _ = reranker.rerank(query, candidate_chunks, seq_k1=seq_k1, seq_k2=seq_k2)
    else:
        ranked_docs, _, _ = reranker.rerank(query, candidate_chunks)

    # Build prompt
    context_block = "\n---\n".join(ranked_docs[:seq_k2])
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. Use the provided context to answer the user question. "
            "If the answer is not contained in the context, reply that you don't have enough information."
        ),
    }
    user_msg = {
        "role": "user",
        "content": f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:",
    }

    try:
        answer = call_mistral([system_msg, user_msg])
    except Exception as e:
        answer = f"LLM generation failed: {e}"

    return jsonify({"answer": answer})

# ---------------------------------------------------------------------------
# Dev entry‑point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        initialize_index()
    except Exception as exc:
        print(f"[WARN] Index initialisation skipped: {exc}")

    app.run(host="0.0.0.0", port=5000, debug=True)
