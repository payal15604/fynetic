import argparse
from pathlib import Path
from config import RAW_DIR, SYSTEM_PKL, FAISS_INDEX_PATH, LOGS_DIR
from src.utils import setup_logger, read_json
from src.rag_system import FinanceRAG

logger = setup_logger("Main", LOGS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Finance RAG - Process JSON filings and build index")
    parser.add_argument("--infile", action="append", help="Path(s) to raw filing JSON(s) in data/raw", required=True)
    parser.add_argument("--save", action="store_true", help="Persist model + FAISS to disk")
    args = parser.parse_args()

    rag = FinanceRAG()
    total_chunks = 0
    for rel in args.infile:
        path = Path(rel)
        if not path.is_absolute():
            path = RAW_DIR / rel
        if not path.exists():
            logger.error(f"Raw JSON not found: {path}")
            continue
        count = rag.process_filing_json(path)
        total_chunks += count

    logger.info(f"Processed total chunks: {total_chunks}")
    if args.save:
        rag.save(SYSTEM_PKL, FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()
