# Notebook 03 — Pinecone Vector Database Upload

## Purpose

Embeds all processed text chunks using OpenAI embeddings and upserts them into a Pinecone serverless vector index. LangSmith tracing is enabled throughout so every embedding batch and retrieval test is observable in the LangSmith dashboard.

## Pipeline Position

```
00_video_extractor ► 01_text_processing ► 02_quality_analysis ► [03_pinecone_upload] ► 04_agent_evaluation ► 05_gradio_deployment
```

---

## What It Does

1. Loads all `chunks_*.json` files from `../data/processed/`
2. Initialises OpenAI embeddings (`text-embedding-3-small`, 1536 dimensions) via LangChain
3. Creates the Pinecone index if it does not exist (serverless, cosine similarity)
4. Cleans chunk metadata — replaces all `None` values with empty strings or `0` so Pinecone accepts them
5. Converts each chunk to a LangChain `Document` with rich metadata
6. Upserts documents to Pinecone in configurable batches with progress tracking
7. Runs post-upload retrieval validation queries
8. Saves an ingestion manifest JSON to disk
9. All operations are traced in **LangSmith**

---

## Prerequisites

### Python Packages

```bash
pip install langchain langchain-openai langchain-pinecone pinecone-client python-dotenv langsmith pandas tqdm numpy matplotlib
```

### API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=ls__...
PINECONE_INDEX_NAME=youtube-rag-mechanical-engineering
PINECONE_NAMESPACE=efficient-engineer-v3
```

> Get your Pinecone API key at [app.pinecone.io](https://app.pinecone.io)  
> Get your LangSmith API key at [smith.langchain.com](https://smith.langchain.com)

### Input

All files matching `../data/processed/chunks_*.json` — produced by **Notebook 01** and validated by **Notebook 02**.

---

## Configuration

All settings are in **Section 2 — Configuration**:

| Variable | Default | Description |
|---|---|---|
| `PROCESSED_DIR` | `../data/processed` | Source chunk files |
| `INDEX_NAME` | `youtube-rag-mechanical-engineering` | Pinecone index name (lowercase, hyphens only) |
| `NAMESPACE` | `efficient-engineer-v3` | Namespace within the index |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Must match the model |
| `PINECONE_CLOUD` | `aws` | Cloud provider for serverless index |
| `PINECONE_REGION` | `us-east-1` | Region for serverless index |
| `BATCH_SIZE` | `100` | Chunks per upsert batch |

> **Cost note:** `text-embedding-3-small` costs $0.02 per million tokens. For ~65 videos with ~4800 chunks of ~700 chars each, expect roughly $0.05–0.10 total.

---

## How to Run

1. Ensure `.env` is populated with all four keys
2. Ensure `../data/processed/` contains chunk files (run NB01 and NB02 first)
3. Open `03_pinecone_vector_database_all_videos.ipynb`
4. Run all cells in order (Sections 1 → 8)

The notebook is **idempotent** — re-running it upserts the same vectors again (Pinecone deduplicates by vector ID), so it is safe to re-run after adding new videos.

---

## LangChain Document Schema

Each chunk is converted to:

```python
Document(
    page_content="Friction is a force that resists...",
    metadata={
        "chunk_id":      "abc123XYZ89_chunk_000",
        "video_id":      "abc123XYZ89",
        "video_title":   "Introduction to Friction",
        "channel":       "Efficient Engineer",
        "url":           "https://www.youtube.com/watch?v=abc123XYZ89",
        "start_time":    12,
        "end_time":      43,
        "start_time_str": "00:12",
        "end_time_str":   "00:43",
        "upload_date":   "20230415",
    }
)
```

All `None` values are replaced before upsert to satisfy Pinecone's metadata requirements.

---

## Pinecone Index Settings

| Setting | Value |
|---|---|
| Metric | Cosine similarity |
| Dimensions | 1536 |
| Type | Serverless |
| Cloud | AWS |
| Region | us-east-1 |

> If you change the embedding model (e.g. to `text-embedding-3-large` at 3072 dims), you must update `EMBEDDING_DIMENSIONS` and create a new index — Pinecone indexes are dimension-locked.

---

## LangSmith Tracing

The notebook sets:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "youtube-rag-engineering-chatbot"
```

All embedding calls and retrieval tests appear in your LangSmith project dashboard automatically.

---

## Output

- Vectors upserted to Pinecone index
- `../data/ingestion_manifest.json` — records which chunks were uploaded, when, and with what config

---

## Common Issues

| Problem | Fix |
|---|---|
| `AuthenticationError` | Check `OPENAI_API_KEY` in `.env` |
| `PineconeException: INVALID_ARGUMENT` | A metadata field contains `None`. Re-run — the notebook cleans these automatically. |
| Index dimension mismatch | You changed the embedding model. Delete the old index in Pinecone console and re-run. |
| `langsmith` import error | `pip install langsmith` |
| Upsert stops mid-way | Re-run — idempotent. Already-uploaded vectors are overwritten harmlessly. |

---

## Next Step

Once the Pinecone index is populated, proceed to:

**`04_rag_agent_and_evaluation.ipynb`** — builds the conversational RAG agent and evaluates it with LangSmith.
