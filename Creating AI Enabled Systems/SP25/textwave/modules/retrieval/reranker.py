import os
import pickle

from sympy import vectorize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import torch
import numpy as np
import os
import pickle


class Reranker:
    """
    Perform reranking of documents based on their relevance to a given query.

    Supports multiple reranking strategies:
    - Cross‑encoder: Uses a transformer model to compute pair‑wise relevance.
    - TF‑IDF: Uses term frequency‑inverse document frequency with similarity metrics.
    - BoW: Uses a Bag‑of‑Words representation with similarity metrics.
    - Hybrid: Combines TF‑IDF and cross‑encoder scores.
    - Sequential: Applies TF‑IDF first, then cross‑encoder for refined reranking.
    """

    def __init__(
        self,
        type,
        cross_encoder_model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    ):
        """
        :param type: One of {"cross_encoder", "tfidf", "bow", "hybrid", "sequential"}
        :param cross_encoder_model_name: HuggingFace model name for the cross‑encoder.
        """
        self.type = type
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(
            cross_encoder_model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)

    # --------------------------------------------------------------------- #
    # Public dispatcher
    # --------------------------------------------------------------------- #
    def rerank(
        self,
        query: str,
        context: list[str],
        distance_metric: str = "cosine",
        seq_k1: int | None = None,
        seq_k2: int | None = None,
    ):
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        if self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric)
        if self.type == "bow":
            return self.bow_rerank(query, context, distance_metric)
        if self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric)
        if self.type == "sequential":
            return self.sequential_rerank(
                query, context, seq_k1, seq_k2, distance_metric
            )
        raise ValueError(f"Unsupported reranker type: {self.type}")

    # --------------------------------------------------------------------- #
    # 1. Cross‑encoder
    # --------------------------------------------------------------------- #
    def cross_encoder_rerank(self, query: str, context: list[str]):
        """Return docs sorted by the raw cross‑encoder relevance logit."""
        if not context:
            return [], [], []

        pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits.squeeze()

        scores = logits.tolist()
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        ranked_docs = [context[i] for i in sorted_idx]
        ranked_scores = [scores[i] for i in sorted_idx]
        return ranked_docs, sorted_idx, ranked_scores

    # --------------------------------------------------------------------- #
    # 2. TF‑IDF
    # --------------------------------------------------------------------- #
    def _vector_distance_rerank(
        self,
        query: str,
        context: list[str],
        vectorizer,
        distance_metric: str = "cosine",
    ):
        if not context:
            return [], [], []

        matrix = vectorizer.fit_transform([query] + context)
        q_vec = matrix[0]
        doc_vecs = matrix[1:]

        distances = pairwise_distances(q_vec, doc_vecs, metric=distance_metric).flatten()
        # Convert to similarity (higher is better). For cosine, similarity=1‑distance.
        if distance_metric == "cosine":
            similarities = 1 - distances
        else:
            similarities = -distances  # smaller distance ⇒ larger similarity

        sorted_idx = np.argsort(similarities)[::-1]
        ranked_docs = [context[i] for i in sorted_idx]
        ranked_scores = similarities[sorted_idx].tolist()
        return ranked_docs, sorted_idx.tolist(), ranked_scores

    def tfidf_rerank(self, query, context, distance_metric: str = "cosine"):
        """Lexical rerank using TF‑IDF vectors."""
        vectorizer = TfidfVectorizer()
        return self._vector_distance_rerank(query, context, vectorizer, distance_metric)

    # --------------------------------------------------------------------- #
    # 3. Bag‑of‑Words
    # --------------------------------------------------------------------- #
    def bow_rerank(self, query, context, distance_metric: str = "cosine"):
        """Lexical rerank using raw term counts (BoW)."""
        vectorizer = CountVectorizer()
        return self._vector_distance_rerank(query, context, vectorizer, distance_metric)

    # --------------------------------------------------------------------- #
    # 4. Hybrid (TF‑IDF + Cross‑encoder)
    # --------------------------------------------------------------------- #
    def hybrid_rerank(
        self,
        query,
        context,
        distance_metric: str = "cosine",
        tfidf_weight: float = 0.3,
    ):
        if not context:
            return [], [], []

        # 1) TF‑IDF similarity
        _, _, tfidf_scores = self.tfidf_rerank(query, context, distance_metric)
        tfidf_scores = np.array(tfidf_scores)

        # 2) Cross‑encoder scores
        _, _, ce_scores = self.cross_encoder_rerank(query, context)
        ce_scores = np.array(ce_scores)

        # Normalise both to [0,1]
        def minmax(x):
            return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.ones_like(x)

        tfidf_norm = minmax(tfidf_scores)
        ce_norm = minmax(ce_scores)

        combined = tfidf_weight * tfidf_norm + (1 - tfidf_weight) * ce_norm
        sorted_idx = np.argsort(combined)[::-1]

        ranked_docs = [context[i] for i in sorted_idx]
        ranked_scores = combined[sorted_idx].tolist()
        return ranked_docs, sorted_idx.tolist(), ranked_scores

    # --------------------------------------------------------------------- #
    # 5. Sequential (k‑nearest TF‑IDF → Cross‑encoder)
    # --------------------------------------------------------------------- #
    def sequential_rerank(
        self,
        query,
        context,
        seq_k1: int | None = None,
        seq_k2: int | None = None,
        distance_metric: str = "cosine",
    ):
        if not context:
            return [], [], []

        # Stage 1 ‑ TF‑IDF
        tfidf_docs, tfidf_idx, tfidf_scores = self.tfidf_rerank(
            query, context, distance_metric
        )
        k1 = seq_k1 or len(context)
        topk1_idx = tfidf_idx[:k1]
        topk1_docs = [context[i] for i in topk1_idx]

        # Stage 2 ‑ Cross‑encoder on the reduced set
        ce_docs, ce_sub_idx, ce_scores = self.cross_encoder_rerank(query, topk1_docs)
        k2 = seq_k2 or len(topk1_docs)

        final_idx = [topk1_idx[i] for i in ce_sub_idx[:k2]]
        final_docs = [context[i] for i in final_idx]
        final_scores = ce_scores[:k2]
        return final_docs, final_idx, final_scores


if __name__ == "__main__":
    query = "What are the health benefits of green tea?"
    docs = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function.",
    ]

    for mode in ["tfidf", "bow", "cross_encoder", "hybrid", "sequential"]:
        print(f"\n=== {mode.upper()} ===")
        r = Reranker(type=mode)
        ranked_docs, idx, scores = r.rerank(query, docs, seq_k1=3, seq_k2=2)
        for rank, (d, s) in enumerate(zip(ranked_docs, scores), 1):
            print(f"Rank {rank}: {s:.4f} | {d}")
