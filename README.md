# **Fynetic**

**RAG-powered LLM platform for corporate financial filings**

---

## Origins

While reading a Forbes article [*How Businesses Can Succeed with LLMs*](https://www.forbes.com/councils/forbestechcouncil/2023/08/23/how-businesses-can-succeed-with-llms/), I realized the tremendous potential of LLMs for business intelligence. Recently I had been reading about Investment Banks and wondered if there was an easier way to go through a company’s financial health? Sifting through dense SEC filings (basically, periodic corporate financial reports) felt overwhelming. 

**What if we could ask questions about a company's financials the same way you'd ask an LLM?**

That’s how **Fynetic** came to life — built to deliver clear, accurate answers directly from filings.

---

## What Fynetic Does

1. **Efficiently retrieves relevant sections** from long financial documents using semantic search.
2. **Generates natural, concise answers**, grounded in real filing content.
3. Powered by **RAG**, combining retrieval precision with generative language models.

---

## How It Works

![How It Works](images/image.png)

### Step 1: Fetch and Chunk

- Integrates with the **SEC EDGAR API** (requires authorized server for headers - recommended Google Servers).
- Splits lengthy filings into digestible chunks.

### Step 2: Embed & Index

- Uses a transformer embedding model ( `all-MiniLM-L6-v2`) to vectorize chunks.
- Stores embeddings in FAISS (Vcetor Database) for fast similarity search.

### Step 3: Query

- Converts your question into an embedding.
- Retrieves top-K matching chunks from the index.

### Step 4: Answer Generation

- Passes retrieved content into an LLM (default: Flan-T5-small).
- Truncates to fit the token limit, then outputs a clean, intelligible answer.

---

## Features at a Glance

- **Accurate retrieval** via embeddings from filing context.
- **Native, plain-English summaries**, enhanced by LLM.
- **Fully configurable**: model, temperature, retrieved chunk count.
- **Graceful truncation** prevents overflow while retaining relevance.
- **Extension-ready**: future support for summarization, keyword weighting, web UI, and LLM upgrades.

---

## Installation & Usage

```bash
git clone https://github.com/yourusername/fynetic.git
cd fynetic
python -m venv fin-env
source fin-env/bin/activate       # or Windows equivalent
pip install -r requirements.txt

```

Set your API key securely (e.g., `GOOGLE_API_KEY`) before running.

```bash
python query_cli.py --q "Who is the CEO of Tesla?" --k 3

=== Answer ===
Tesla's CEO is Elon Musk, as reported in the latest filing.

=== Sections Retrieved ===
1. ITEM 11 — Executive Officers (score: 0.341)
2. ITEM 8  — Management Discussion (score: 0.327)
3. ITEM 1A — Risk Factors (score: 0.312)

```

You'll get both the answer and the relevant filing sections returned to you.

---

## Models Used

| Component | Model Used | Dimension | Purpose |
| --- | --- | --- | --- |
| **Embedder** | `sentence-transformers/all-MiniLM-L6-v2` | 768 | Embedding filings + queries |
| **LLM (Default)** | `google/flan-t5-small` | 768 | Lightweight summarization |
| **LLM (Optional)** | `gemini-2.5-flash-lite-preview-06-17` | 1M | Better contextual summaries |

---

## Why It Stands Out

- Inspired by real-world business insights turned into actionable AI tool.
- Context-aware, accurate, and easy to use.
- Scalable and ready for future improvements — AI applied wisely, where it matters most.

---

## **Future Roadmap** 🛠️

- **UX Overhaul** → Shift from CLI to a **clean web-based interface**
- **Larger LLMs** → Support GPT-4 / Gemini-Pro for better financial insights
- **Keyword-Aware Retrieval** → Prioritize domain-specific terminology in vector search
- **Multi-Filing Contexts** → Analyze multiple companies for competitive benchmarking

