import argparse
from config import SYSTEM_PKL, FAISS_INDEX_PATH, LOGS_DIR, TOP_K
from src.utils import setup_logger
from src.rag_system import FinanceRAG
from src.query_interface import FinanceRAGQuery

logger = setup_logger("CLI", LOGS_DIR)

def main():
    parser = argparse.ArgumentParser(description="Finance RAG - Query CLI")
    parser.add_argument("--q", required=True, help="Your question")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top K")
    parser.add_argument("--no-llm", action="store_true", help="Disable HF LLM")
    args = parser.parse_args()

    rag = FinanceRAG.load()
    print('RAG system loaded.')
    qi = FinanceRAGQuery(rag, use_llm=not args.no_llm)

    result = qi.ask(args.q, k=args.k)
    print("Question Asked")
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sections ===\n")
    for i, s in enumerate(result["sections"], 1):
        print(f"{i}. {s['section']} — {s['description']} (score: {s['score']:.3f})")
    print()

if __name__ == "__main__":
    main()
