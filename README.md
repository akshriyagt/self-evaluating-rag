# Self-Evaluating RAG Pipeline

A Retrieval-Augmented Generation pipeline that doesn't just answer questions —
it **checks its own answers** against retrieved context and flags low-confidence
or unsupported responses instead of confidently hallucinating.

Built to run entirely on free tiers: no GPU, no paid infra required.

## Why this project is different from a basic RAG tutorial

Most RAG demos stop at "retrieve chunks -> generate answer." This one adds:

1. **A judge layer** — a second LLM call that scores whether the generated
   answer is actually grounded in the retrieved context (faithfulness score).
2. **Confidence-based flagging** — low-grounding answers are labeled
   "uncertain" instead of returned as fact.
3. **An evaluation harness** — a hand-built test set of answerable and
   deliberately unanswerable questions, scored for precision/recall on
   "correctly says I don't know."
4. **A monitoring dashboard** — every query, retrieval, and confidence score
   is logged and viewable in a simple Streamlit UI.

## Architecture

```
Documents -> Chunker -> Embeddings (MiniLM, CPU) -> Chroma vector store
                                                          |
User Query -> Embed query -> Retrieve top-k chunks -> LLM generates answer
                                                          |
                                        Judge LLM call: "Is this grounded?"
                                                          |
                                   Confidence score -> Answer or "uncertain"
                                                          |
                                        Logged -> Streamlit dashboard
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

This uses the Anthropic API for generation and judging (you can swap in
OpenAI by editing `src/rag_pipeline.py` — the call sites are isolated).

```bash
cp .env.example .env
# edit .env and paste your API key
```

Get a free/low-cost API key at https://console.anthropic.com

### 3. Add your documents

Drop `.txt` or `.md` files into `sample_docs/`. A sample document is already
included so you can test immediately without your own data.

### 4. Ingest documents (builds the vector store)

```bash
python src/ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

This opens a browser UI where you can ask questions and see:
- The generated answer
- The confidence/faithfulness score
- The retrieved chunks it was based on
- A running log of all queries (the "monitoring dashboard")

### 6. Run the evaluation harness

```bash
python eval/run_eval.py
```

This runs a fixed set of test questions (some answerable from the docs, some
not) and reports precision/recall on correctly detecting unanswerable
questions — the core metric that proves your "I don't know" logic works.

## Project structure

```
rag-pipeline/
├── app.py                  # Streamlit UI + dashboard
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py            # Chunking + embedding + vector store build
│   ├── rag_pipeline.py       # Retrieval + generation + judge logic
│   └── judge.py              # Faithfulness scoring logic
├── eval/
│   ├── eval_questions.json   # Hand-crafted test set
│   └── run_eval.py           # Runs eval, prints precision/recall
├── sample_docs/
│   └── sample.txt             # Example document to test with
└── chroma_db/                 # Local vector store (created on ingest)
```

## What to put in your portfolio / resume writeup

- "Built a RAG pipeline with an automated faithfulness-checking layer that
  reduces hallucinated answers by flagging ungrounded responses."
- "Designed and ran an evaluation harness measuring precision/recall on
  answerability detection across N test queries."
- "Implemented full observability: query logging, confidence scoring, and a
  monitoring dashboard for production-style visibility."

Push this repo to GitHub, replace `sample_docs/` with a real document set
(company handbook, research papers, product docs — whatever is relevant to
the job you're targeting), and record a 2-minute demo video for extra
impact.

## Extending this further (optional, for later)

- Swap Chroma for a hosted vector DB (Pinecone/Weaviate free tier) to show
  cloud deployment experience.
- Add a cost/latency tracker per query.
- Add a second judge check for retrieval quality separate from answer
  faithfulness (did we even retrieve the right chunks?).
- Containerize with Docker for a "production-ready" feel.
