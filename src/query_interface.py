from typing import Dict, List
from config import HF_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LOGS_DIR
from .rag_system import FinanceRAG
from .utils import setup_logger

logger = setup_logger("Query", LOGS_DIR)

try:
    from transformers import pipeline
    HAS_HF = True
except Exception:
    HAS_HF = False

# class FinanceRAGQuery:
#     def __init__(self, rag: FinanceRAG, use_llm: bool = True):
#         self.rag = rag
#         self.use_llm = use_llm and HAS_HF and HF_MODEL_NAME is not None
#         self.llm = None
#         if self.use_llm:
#             try:
#                 logger.info(f"Loading HF model: {HF_MODEL_NAME}")
#                 self.llm = pipeline("text2text-generation", model=HF_MODEL_NAME)
#             except Exception as e:
#                 logger.warning(f"LLM load failed: {e}")
#                 self.use_llm = False

#     def ask(self, question: str, k: int = 5) -> Dict:
#         results = self.rag.search(question, k=k)
#         filing_context = self._format(results)
#         if self.use_llm and filing_context:
#             answer = self._llm_answer(question, filing_context)
#         else:
#             answer = f"Retrieved {len(results)} sections.\n\n" + filing_context[:1200]
#         return {
#             "question": question,
#             "answer": answer,
#             "sections": [
#                 {"section": r["section"], "description": r["description"], "text": r["text"], "score": r["score"]}
#                 for r in results
#             ]
#         }

#     def _format(self, results: List[Dict]) -> str:
#         if not results:
#             return ""
#         parts = []
#         for i, r in enumerate(results, 1):
#             parts.append(f"[{i}] {r['section']} — {r['description']} (score: {r['score']:.3f})\n{r['text']}\n")
#         return "\n".join(parts)

#     def _llm_answer(self, q: str, ctx: str) -> str:
#         prompt = (
#             "You are answering questions using excerpts from SEC filings. "
#             "Use only the provided context. Cite the section titles when possible.\n\n"
#             f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
#         )
#         try:
#             out = self.llm(
#                 prompt,
#                 max_new_tokens=LLM_MAX_NEW_TOKENS,
#                 temperature=LLM_TEMPERATURE,
#                 do_sample=False
#             )
#             return out[0]["generated_text"].strip()
#         except Exception as e:
#             logger.warning(f"LLM failed: {e}")
#             return f"(No LLM) Context-based answer:\n{ctx[:800]}"

# from transformers import AutoTokenizer

# class FinanceRAGQuery:
#     def __init__(self, rag: FinanceRAG, use_llm: bool = True):
#         self.rag = rag
#         self.use_llm = use_llm and HAS_HF and HF_MODEL_NAME is not None
#         self.llm = None
#         self.tokenizer = None
#         if self.use_llm:
#             try:
#                 logger.info(f"Loading HF model: {HF_MODEL_NAME}")
#                 from transformers import pipeline, AutoTokenizer
#                 self.llm = pipeline("text2text-generation", model=HF_MODEL_NAME)
#                 self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
#             except Exception as e:
#                 logger.warning(f"LLM load failed: {e}")
#                 self.use_llm = False

#     from transformers import AutoTokenizer

#     def _llm_answer(self, q: str, ctx: str) -> str:
#         prompt = (
#             "You are answering questions using excerpts from SEC filings. "
#             "Use only the provided context. Cite the section titles when possible.\n\n"
#             f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
#         )

#         try:
#             # Load tokenizer dynamically for your HF model
#             from transformers import AutoTokenizer
#             tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

#             # Truncate the prompt to model's max tokens
#             max_len = tokenizer.model_max_length or 512
#             tokens = tokenizer.encode(prompt, truncation=True, max_length=max_len)
#             truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)

#             out = self.llm(
#                 truncated_prompt,
#                 max_new_tokens=LLM_MAX_NEW_TOKENS,
#                 temperature=LLM_TEMPERATURE,
#                 do_sample=False
#             )

#             # Handle empty output safely
#             if not out or "generated_text" not in out[0] or not out[0]["generated_text"].strip():
#                 return "(LLM returned empty output. Try reducing context or switching to a bigger model.)"

#             return out[0]["generated_text"].strip()

#         except Exception as e:
#             logger.warning(f"LLM failed: {e}")
#             return f"(No LLM) Context-based answer:\n{ctx[:800]}"

from typing import Dict, List
from config import HF_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE, LOGS_DIR
from .rag_system import FinanceRAG
from .utils import setup_logger
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyA9OVyV8OjVOb8PXCeg02Obd4IFIjIzIEQ")



logger = setup_logger("Query", LOGS_DIR)

try:
    from transformers import pipeline, AutoTokenizer
    HAS_HF = True
except Exception:
    HAS_HF = False


class FinanceRAGQuery:
    def __init__(self, rag: FinanceRAG, use_llm: bool = True):
        self.rag = rag
        self.use_llm = use_llm and HAS_HF and HF_MODEL_NAME is not None
        self.llm = None
        self.tokenizer = None
        if self.use_llm:
            try:
                logger.info(f"Loading HF model: {HF_MODEL_NAME}")
                self.llm = pipeline("text2text-generation", model=HF_MODEL_NAME)
                self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            except Exception as e:
                logger.warning(f"LLM load failed: {e}")
                self.use_llm = False

    def ask(self, question: str, k: int = 5) -> Dict:
        """Main entry point — retrieves top-K chunks & generates an answer"""
        results = self.rag.search(question, k=k)
        filing_context = self._format(results)

        if self.use_llm and filing_context:
            answer = self._llm_answer(question, filing_context)
        else:
            answer = f"Retrieved {len(results)} sections.\n\n" + filing_context[:1200]

        return {
            "question": question,
            "answer": answer,
            "sections": [
                {
                    "section": r["section"],
                    "description": r["description"],
                    "text": r["text"],
                    "score": r["score"]
                }
                for r in results
            ]
        }

    def _format(self, results: List[Dict]) -> str:
        """Nicely formats retrieved chunks for LLM context"""
        if not results:
            return ""
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[{i}] {r['section']} — {r['description']} "
                f"(score: {r['score']:.3f})\n{r['text']}\n"
            )
        return "\n".join(parts)

    def _llm_answer(self, q: str, ctx: str) -> str:
    
        """Generates an LLM-based answer, ensuring prompt length fits the model"""
        prompt = (
            "You are a financial analyst assistant. Answer the user’s question based only on the following excerpts from the company's SEC filings.  "
            "Answer in clear, concise, natural English. Summarize key points and cite the section numbers when possible.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
        )

        try:
            # Truncate the prompt to the model's maximum token limit
            max_len = self.tokenizer.model_max_length or 700
            tokens = self.tokenizer.encode(prompt, truncation=True, max_length=max_len)
            truncated_prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)

            # out = self.llm(
            #     truncated_prompt,
            #     max_new_tokens=LLM_MAX_NEW_TOKENS,
            #     temperature=LLM_TEMPERATURE,
            #     do_sample=True
            # )
            
            out = genai.generate_text(
            model="gemini-2.5-flash-lite-preview-06-17",
            prompt=truncated_prompt)

            # Extract the embedding
            embedding = out[0].get("embedding", None)
            if embedding:
                print(f"Embedding dimension: {len(embedding)}")
            else:
                print("Embedding not found.")

            # Handle edge cases when LLM returns empty or malformed output
            if not out or "generated_text" not in out[0] or not out[0]["generated_text"].strip():
                return "(LLM returned empty output. Try reducing context or switching to a bigger model.)"

            return out[0]["generated_text"].strip()

        except Exception as e:
            logger.warning(f"LLM failed: {e}")
            return f"(No LLM) Context-based answer:\n{ctx[:800]}"
