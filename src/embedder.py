from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

class EmbeddingModel:
    def __init__(self, model_name: str):
        if not HAS_ST:
            raise ImportError("sentence-transformers not installed. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embs, dtype=np.float32)
