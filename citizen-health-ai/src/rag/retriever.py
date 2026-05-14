"""
TF-IDF retriever for NHIM knowledge base — pure numpy, no sklearn needed.

Builds a term-document TF-IDF matrix at startup, then uses cosine similarity
to find the most relevant scheme documents for each user query.
"""

import math
import re
from collections import Counter

import numpy as np

from .nhim_knowledge_base import NHIM_DOCUMENTS


class NHIMRetriever:
    def __init__(self, documents: list[dict]) -> None:
        self._docs = documents
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._matrix: np.ndarray
        self._build_index()

    # ------------------------------------------------------------------ #
    #  Index construction                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # Keep alphanumeric tokens; split on anything else
        return re.findall(r"[a-z0-9]+", text.lower())

    def _doc_text(self, doc: dict) -> str:
        # Title and tags get extra weight by repeating them
        tags = " ".join(doc.get("tags", []))
        return f"{doc['title']} {doc['title']} {tags} {tags} {doc['content']}"

    def _build_index(self) -> None:
        n = len(self._docs)
        corpus = [self._tokenize(self._doc_text(d)) for d in self._docs]

        # Document frequency for IDF
        df: dict[str, int] = {}
        for tokens in corpus:
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

        # Assign vocab index and IDF score
        self._vocab = {w: i for i, w in enumerate(df)}
        self._idf = {w: math.log((n + 1) / (cnt + 1)) + 1.0 for w, cnt in df.items()}

        # Build TF-IDF matrix (n_docs × vocab_size)
        v = len(self._vocab)
        matrix = np.zeros((n, v), dtype=np.float32)
        for row, tokens in enumerate(corpus):
            tf = Counter(tokens)
            total = len(tokens) or 1
            for word, count in tf.items():
                if word in self._vocab:
                    col = self._vocab[word]
                    matrix[row, col] = (count / total) * self._idf[word]

        # L2-normalise rows for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    # ------------------------------------------------------------------ #
    #  Query                                                                #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
        """
        Return top_k documents most relevant to `query`.

        Each returned dict is the original document augmented with a 'score' key.
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []

        v = len(self._vocab)
        q_vec = np.zeros(v, dtype=np.float32)
        tf = Counter(tokens)
        total = len(tokens)
        for word, count in tf.items():
            if word in self._vocab:
                col = self._vocab[word]
                q_vec[col] = (count / total) * self._idf.get(word, 1.0)

        norm = float(np.linalg.norm(q_vec))
        if norm == 0:
            return []
        q_vec /= norm

        scores = self._matrix @ q_vec  # shape (n_docs,)
        top_idx = int(min(top_k, len(self._docs)))
        ranked = np.argsort(scores)[-top_idx:][::-1]

        return [
            {**self._docs[i], "score": float(scores[i])}
            for i in ranked
            if float(scores[i]) >= min_score
        ]

    def format_context(self, query: str, top_k: int = 3) -> str:
        """
        Return retrieved documents as a formatted string for LLM injection.
        Returns empty string if nothing relevant is found.
        """
        docs = self.retrieve(query, top_k=top_k)
        if not docs:
            return ""

        parts = ["=== Relevant NHIM Scheme Information ==="]
        for doc in docs:
            parts.append(f"\n**{doc['title']}**\n{doc['content']}")
        parts.append("\n=== End of NHIM context ===")
        return "\n".join(parts)


# Module-level singleton — built once at import time
nhim_retriever = NHIMRetriever(NHIM_DOCUMENTS)
