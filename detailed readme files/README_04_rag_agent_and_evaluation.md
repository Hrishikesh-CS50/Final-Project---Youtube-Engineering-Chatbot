# Notebook 04 — RAG Agent Build & LangSmith Evaluation

## Purpose

Builds a production-grade conversational RAG agent backed by the Pinecone vector index, then evaluates it using LangSmith's LLM-as-judge framework. At the end of the notebook an `agent_config.json` is exported — this file is consumed by Notebook 05 and `app.py`.

## Pipeline Position

```
00_video_extractor ► 01_text_processing ► 02_quality_analysis ► 03_pinecone_upload ► [04_agent_evaluation] ► 05_gradio_deployment
```

---

## What It Does

### Part 1 — Agent Build

1. Initialises LangChain components: `ChatOpenAI`, `OpenAIEmbeddings`, `PineconeVectorStore`
2. Defines **three tools** the agent can call:
   - `search_transcripts` — semantic search over transcript chunks (core RAG tool)
   - `get_video_info` — returns full metadata for a given video ID
   - `find_videos` — searches for videos covering a specific topic
3. Builds a structured system prompt that constrains the agent to answer only from transcript evidence
4. Creates an `AgentExecutor` using `create_openai_functions_agent` with LangGraph `MemorySaver` for multi-turn memory
5. Runs a single-question smoke test
6. Runs a multi-turn conversation test (3 questions with history)
7. Exports `agent_config.json` to disk

### Part 2 — LangSmith Evaluation

8. Wraps the agent in a `predict_rag_answer()` function compatible with LangSmith's evaluator API
9. Creates a LangSmith evaluation dataset with domain-specific test questions and reference answers
10. Runs LLM-as-judge evaluators measuring: **accuracy**, **hallucination**, **relevance**, and **helpfulness**
11. Results appear in the LangSmith dashboard

---

## Prerequisites

### Python Packages

```bash
pip install langchain langchain-openai langchain-pinecone langgraph langsmith python-dotenv rank-bm25 pydantic
```

### API Keys

All four keys must be in `.env`:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=ls__...
PINECONE_INDEX_NAME=youtube-rag-mechanical-engineering
PINECONE_NAMESPACE=efficient-engineer-v3
```

### Input

A populated Pinecone index — produced by **Notebook 03**.

---

## Configuration

Settings are in **Section 2 — Configuration**:

| Variable | Default | Description |
|---|---|---|
| `INDEX_NAME` | `youtube-rag-mechanical-engineering` | Must match NB03 |
| `NAMESPACE` | `efficient-engineer-v3` | Must match NB03 |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM for the agent |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match NB03 |
| `TOP_K` | `5` | Number of chunks retrieved per query |

> **Cost control:** The notebook defaults to `gpt-4o-mini`. Switch to `gpt-4o` only if you need higher reasoning quality — it costs ~15× more per token.

---

## Agent Tools

### `search_transcripts(query: str) → str`
Runs a cosine similarity search in Pinecone and returns the top-K matching transcript chunks formatted with video title, timestamp, and text. This is the primary RAG retrieval tool.

### `get_video_info(video_id: str) → str`
Returns structured metadata (title, channel, duration, URL, upload date) for a given video ID. Useful when the user asks "tell me more about this video".

### `find_videos(topic: str) → str`
Searches for unique videos in the index that cover a given topic. Returns a deduplicated list of titles and URLs.

---

## System Prompt

The agent is instructed to:
- Answer **only** from transcript search results — no hallucination from outside knowledge
- Always cite sources with video title and timestamp
- Suggest relevant videos when appropriate
- Clearly state when the transcripts do not contain enough information

---

## Multi-Turn Memory

The agent uses LangGraph's `MemorySaver` for in-session memory. Conversation history is passed as `chat_history` in each `AgentExecutor.invoke()` call. History persists within a Python session but resets on kernel restart — for persistent sessions see `app.py`.

---

## LangSmith Evaluation

The evaluation dataset includes questions like:

```
"What is tensile strength?"
"What is the coefficient of friction?"
"Explain stress vs strain"
"What causes fatigue failure in metals?"
```

Each evaluator (accuracy, hallucination, relevance, helpfulness) calls GPT-4o to score the agent's answer against the reference on a 1–5 scale. Results are streamed to the LangSmith dashboard in real time.

---

## Output

### `agent_config.json`

Saved to the project root (or `../`), consumed by NB05 and `app.py`:

```json
{
  "created_at": "2026-03-13T10:00:00",
  "chat_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "index_name": "youtube-rag-mechanical-engineering",
  "namespace": "efficient-engineer-v3",
  "langchain_version": "0.3.x",
  "langsmith_project": "youtube-rag-engineering-chatbot"
}
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `VectorStore not found` | Run Notebook 03 first — the Pinecone index must exist |
| Agent returns "I don't know" | The query may not match any chunks. Try broader phrasing or check `TOP_K`. |
| `MemorySaver` import error | `pip install langgraph` |
| LangSmith evaluation costs too much | Reduce the evaluation dataset size or switch judge model to `gpt-4o-mini` |
| `max_iterations` exceeded | The agent is looping. Increase `max_iterations` in `AgentExecutor` or simplify the question. |

---

## Next Step

Once `agent_config.json` is saved, proceed to:

**`05_gradio_deployment.ipynb`** — tests the Gradio interface in the notebook before launching `app.py`.
