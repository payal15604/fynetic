import os
from pathlib import Path

# base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # put your pre-downloaded filings (json) here
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = DATA_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# files
SECTION_MAP_PATH = CONFIG_DIR / "section_map.json"
SYSTEM_PKL = MODELS_DIR / "rag_system.pkl"
FAISS_INDEX_PATH = MODELS_DIR / "rag_system_faiss.index"

# embedding / search
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
#"sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 32

# chunking
CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 80

# retrieval
TOP_K = 5

# llm (optional; only used by query_interface)
HF_MODEL_NAME = "google/flan-t5-small"  # can set to None to skip
LLM_MAX_NEW_TOKENS = 400
LLM_TEMPERATURE = 0.6
