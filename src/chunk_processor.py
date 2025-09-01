from typing import List
from sentence_transformers import SentenceTransformer
import re

class Chunk:
    def __init__(self, text: str, doc_id: str, chunk_id: int):
        self.text = text
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.summary = None
        self.keywords = None
        self.embedding = None

class ChunkProcessor:
    def __init__(self, summarizer=None, keyword_extractor=None):
        self.summarizer = summarizer
        self.keyword_extractor = keyword_extractor

    def process(self, chunk: Chunk, max_summary_tokens=100) -> Chunk:
        # Summarization
        if self.summarizer:
            try:
                chunk.summary = self.summarizer.summarize(chunk.text, max_tokens=max_summary_tokens)
            except Exception:
                chunk.summary = None
        
        # Keyword extraction
        if self.keyword_extractor:
            try:
                chunk.keywords = self.keyword_extractor.extract(chunk.text)
            except Exception:
                chunk.keywords = None
        
        return chunk

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, chunk: Chunk) -> Chunk:
        # Prefer summary + keywords for embedding
        text_to_embed = chunk.summary or chunk.text
        if chunk.keywords:
            text_to_embed += " " + " ".join(chunk.keywords)
        chunk.embedding = self.model.encode([text_to_embed])[0]
        return chunk
