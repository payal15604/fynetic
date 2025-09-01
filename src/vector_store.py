from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

class VectorStore:
    """
    FAISS inner-product index with L2-normalized vectors.
    Fallback to NumPy cosine if FAISS unavailable.
    """
    def __init__(self):
        self.index = None
        self.embeddings = None  # fallback storage
        self.metadata: List[Dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.size == 0:
            return
        self.metadata = metadata
        if HAS_FAISS:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            self.embeddings = None
        else:
            self.index = None
            self.embeddings = embeddings  # keep for numpy search

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        if embeddings.size == 0:
            return
        if HAS_FAISS and self.index is not None:
            self.index.add(embeddings)
        else:
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        self.metadata.extend(metadata)

    def search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if HAS_FAISS and self.index is not None:
            #scores, idxs = self.index.search(query_emb, k)
            query_emb = np.array(query_emb).astype("float32")

            # Print the dimensions for debugging
            print(f"FAISS index expected dim: {self.index.d}")
            print(f"Query embedding dim: {query_emb.shape[1] if len(query_emb.shape) > 1 else query_emb.shape[0]}")

            # Now perform search
            scores, idxs = self.index.search(query_emb, k)

            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
        # fallback: cosine via dot since embeddings normalized
        if self.embeddings is None or self.embeddings.size == 0:
            return []
        sims = self.embeddings @ query_emb[0].T  # (N,)
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]
