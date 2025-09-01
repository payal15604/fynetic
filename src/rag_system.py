import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from config import (
    EMBEDDING_MODEL_NAME, EMBED_BATCH_SIZE,
    CHUNK_WORDS, CHUNK_OVERLAP_WORDS,
    SYSTEM_PKL, FAISS_INDEX_PATH, MODELS_DIR, LOGS_DIR
)
from .utils import setup_logger, read_json, write_json, ensure_dir
from .parser_sec import parse_filing_record
from .chunker import chunk_by_words_preserving_sentences
from .embedder import EmbeddingModel
from .vector_store import VectorStore, HAS_FAISS

logger = setup_logger("RAG", LOGS_DIR)

@dataclass
class Chunk:
    company: str
    cik: str
    filing_type: str
    filing_date: str
    section: str
    description: str
    item_number: str
    chunk_index: int
    text: str

class FinanceRAG:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.encoder = EmbeddingModel(model_name)
        self.vs = VectorStore()
        self.chunks: List[Chunk] = []
        ensure_dir(MODELS_DIR)

    def process_filing_json(self, path: Path):
        raw = read_json(path)

        # ✅ CASE 1 — Already pre-chunked JSON
        if "chunks" in raw and isinstance(raw["chunks"], list):
            pre_chunks = []
            for c in raw["chunks"]:
                if "text" not in c or not c["text"].strip():
                    continue
                pre_chunks.append(Chunk(
                    company=c.get("company", "Unknown"),
                    cik=c.get("cik", "Unknown"),
                    filing_type=c.get("filing_type", "Unknown"),
                    filing_date=c.get("filing_date", "Unknown"),
                    section=c.get("section", "Unknown"),
                    description=c.get("description", c.get("section", "Unknown")),
                    item_number=c.get("item_number", ""),
                    chunk_index=c.get("chunk_index", 0),
                    text=c["text"]
                ))

            if not pre_chunks:
                logger.warning(f"No valid chunks found in {path.name}")
                return 0

            # Embed & add to vector store
            texts = [c.text for c in pre_chunks]
            embs = self.encoder.encode(texts, batch_size=EMBED_BATCH_SIZE)
            self._add_embeddings(pre_chunks, embs)
            logger.info(f"Loaded {len(pre_chunks)} pre-chunked chunks from {path.name}")
            return len(pre_chunks)

        # ✅ CASE 2 — Raw SEC filings → Fall back to old logic
        rec = parse_filing_record(raw)
        meta = rec["meta"]
        sections = rec["sections"]

        new_chunks: List[Chunk] = []
        for sec in sections:
            sec_name = sec["section"]
            item = sec.get("item_number", "")
            desc = sec_name.split(" - ", 1)[-1] if " - " in sec_name else "Unknown Section"

            parts = chunk_by_words_preserving_sentences(
                sec["text"], CHUNK_WORDS, CHUNK_OVERLAP_WORDS
            )
            for idx, chunk_text in enumerate(parts):
                new_chunks.append(Chunk(
                    company=meta["company"],
                    cik=meta["cik"],
                    filing_type=meta["filing_type"],
                    filing_date=meta["filing_date"],
                    section=sec_name,
                    description=desc,
                    item_number=item,
                    chunk_index=idx,
                    text=chunk_text
                ))

        if not new_chunks:
            logger.warning(f"No chunks created from {path.name}")
            return 0

        texts = [c.text for c in new_chunks]
        embs = self.encoder.encode(texts, batch_size=EMBED_BATCH_SIZE)
        self._add_embeddings(new_chunks, embs)
        logger.info(f"Processed {path.name}: {len(new_chunks)} chunks")
        return len(new_chunks)


    def _add_embeddings(self, chunks: List[Chunk], embs: np.ndarray):
        self.chunks.extend(chunks)
        meta = [c.__dict__ for c in chunks]
        if self.vs.metadata:
            self.vs.add(embs, meta)
        else:
            self.vs.build(embs, meta)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = self.encoder.encode([query], batch_size=1)  # normalized
        hits = self.vs.search(q, k)
        results = []
        for idx, score in hits:
            m = self.vs.metadata[idx]
            preview = (m["text"][:700] + "…") if len(m["text"]) > 700 else m["text"]
            results.append({
                "text": preview,
                "score": score,
                "section": m["section"],
                "description": m["description"],
                "metadata": {
                    "company": m["company"],
                    "filing_type": m["filing_type"],
                    "filing_date": m["filing_date"],
                    "cik": m["cik"],
                    "item_number": m["item_number"],
                    "chunk_index": m["chunk_index"]
                }
            })
        return results

    def save(self, pkl_path: Path = SYSTEM_PKL, faiss_path: Path = FAISS_INDEX_PATH):
        # save chunks & light metadata; embeddings live only in FAISS/ram
        data = {
            "chunks": [c.__dict__ for c in self.chunks],
            "model_name": EMBEDDING_MODEL_NAME
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        if HAS_FAISS and self.vs.index is not None:
            import faiss
            faiss.write_index(self.vs.index, str(faiss_path))
        logger.info(f"Saved system to {pkl_path}")

    @classmethod
    def load(cls, pkl_path: Path = SYSTEM_PKL, faiss_path: Path = FAISS_INDEX_PATH) -> "FinanceRAG":
        inst = cls()
        if not pkl_path.exists():
            logger.warning(f"{pkl_path} not found; starting empty")
            return inst
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            # rehydrate chunks list
            inst.chunks = [Chunk(**c) for c in data.get("chunks", [])]
            # rebuild embeddings index from scratch is not possible here (we didn't persist embs),
            # so we rely on FAISS index file if present; otherwise search will be empty until reprocess.
            if faiss_path.exists() and HAS_FAISS:
                import faiss
                inst.vs.index = faiss.read_index(str(faiss_path))
                inst.vs.metadata = [c.__dict__ for c in inst.chunks]
                inst.vs.embeddings = None
                logger.info(f"Loaded FAISS index and {len(inst.chunks)} chunks")
            else:
                # no FAISS file; degrade gracefully (must re-run processing to rebuild)
                inst.vs.index = None
                inst.vs.embeddings = None
                inst.vs.metadata = [c.__dict__ for c in inst.chunks]
                logger.warning("No FAISS index file found; search will be empty until you reprocess this session.")
            return inst
        except Exception as e:
            logger.error(f"Failed to load system: {e}")
            return cls()
