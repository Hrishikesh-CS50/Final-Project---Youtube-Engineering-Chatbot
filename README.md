![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project III | Business Case: Building a Multimodal AI ChatBot for YouTube Video QA
# 🔧 YouTube Engineering Chatbot — RAG Pipeline

An end-to-end pipeline that turns YouTube engineering videos into a conversational AI chatbot. Ask questions, get answers grounded in real transcript content, and jump directly to the relevant video timestamp.

> Built with LangChain · OpenAI · Pinecone · LangSmith · Gradio

---

## 🎬 Demo

[![Watch Demo](https://img.shields.io/badge/Watch-Demo%20Video-red?logo=youtube)](./Deployment-Exploring%20the%20Engineering%20R....mp4)
📄 [View Final Presentation](./YouTube_Chatbot_Final_presentation.pdf)

---

## 🗺️ How It Works

```
YouTube URLs
    │
    ▼  [NB00] yt-dlp + youtube-transcript-api
Raw JSON (transcripts + metadata)
    │
    ▼  [NB01] LangChain RecursiveCharacterTextSplitter
Processed chunks (700 chars, 150 overlap, with timestamps)
    │
    ▼  [NB02] Schema · size · duplicate checks  →  PASS / WARN / FAIL
Quality-validated corpus
    │
    ▼  [NB03] OpenAI text-embedding-3-small → Pinecone serverless index
Vector database
    │
    ▼  [NB04] LangChain AgentExecutor + LangGraph MemorySaver + LangSmith eval
Conversational RAG agent
    │
    ▼  [NB05] Gradio UI  →  app.py
Chat interface with timestamped video cards
```

---

## 📁 Repository Structure

```
Final-Project---Youtube-Engineering-Chatbot/
├── notebooks/
│   ├── 00_video_extractor_universal_url.ipynb
│   ├── 01_text_processing_all_videos_timestamped.ipynb
│   ├── 02_corpus_quality_analysis.ipynb
│   ├── 03_pinecone_vector_database_all_videos.ipynb
│   ├── 04_rag_agent_and_evaluation.ipynb
│   └── 05_gradio_deployment.ipynb
├── detailed readme files/          ← per-notebook deep-dive docs
├── data/                           ← auto-created by the pipeline
│   ├── raw/                        ← NB00 output: video_*.json
│   └── processed/                  ← NB01 output: chunks_*.json
├── config/
├── app.py                          ← standalone Gradio app (34 KB)
├── agent_config.json               ← exported by NB04, consumed by NB05 + app.py
├── requirements.txt
├── .env                            ← your API keys (never commit)
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Final-Project---Youtube-Engineering-Chatbot.git
cd Final-Project---Youtube-Engineering-Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env
```

Open `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=ls__...
PINECONE_INDEX_NAME=youtube-rag-mechanical-engineering
PINECONE_NAMESPACE=efficient-engineer-v3
```

| Key | Where to get it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) |

### 5. Run the pipeline notebooks in order

Open Jupyter and run each notebook top-to-bottom:

```
00 → 01 → 02 → 03 → 04 → 05
```

### 6. Launch the app

```bash
python app.py
```

---

## 📓 Notebook Summaries

### `00` — Video Extractor
Downloads transcripts and metadata from any YouTube URL — single video, Short, or full playlist. Saves `data/raw/video_{id}.json` per video. No API keys required.

```python
youtube_url = "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
max_videos  = 5   # use None for the full playlist
```

**Packages:** `yt-dlp`, `youtube-transcript-api`, `tqdm`

---

### `01` — Text Processing
Splits each transcript into overlapping chunks with timestamp metadata. Saves `data/processed/chunks_{id}.json` per video.

| Setting | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | `700` chars | Tuned for `text-embedding-3-small` |
| `CHUNK_OVERLAP` | `150` chars | Keeps context across chunk boundaries |

**Packages:** `langchain`, `langchain-text-splitters`

---

### `02` — Corpus Quality Analysis
Validates every chunk before upload. Checks schema completeness, character counts, timestamp ranges, and duplicates. Prints a **PASS / WARN / FAIL** verdict. No API keys needed.

> ⚠️ Fix all **ERROR**-level issues before running NB03.

---

### `03` — Pinecone Upload
Embeds chunks with OpenAI and upserts to a Pinecone serverless index. Safe to re-run — the notebook is idempotent.

**Estimated cost:** ~$0.05–0.10 for ~65 videos / ~4,800 chunks.

**Packages:** `langchain-openai`, `langchain-pinecone`, `langsmith`

---

### `04` — RAG Agent & Evaluation
Builds a multi-turn conversational agent with three tools and evaluates it with LangSmith LLM-as-judge scoring.

| Tool | What it does |
|---|---|
| `search_transcripts` | Semantic search over transcript chunks (core RAG) |
| `get_video_info` | Returns full metadata for a video ID |
| `find_videos` | Finds videos covering a specific topic |

**Evaluation metrics:** accuracy · hallucination · relevance · helpfulness (1–5 scale, judged by GPT-4o)

**Output:** `agent_config.json`

---

### `05` — Gradio Deployment
Tests the full chat UI inside the notebook before going live. Mirrors `app.py` exactly.

```bash
# Once satisfied with notebook tests:
python app.py

# To share a temporary public URL:
demo.launch(share=True)
```

---

## 🖥️ Chat UI

```
┌──────────────────────────────────────────────────┐
│  🔧 YouTube Engineering Chatbot                  │
├──────────────────────────────────────────────────┤
│  Chat window                                     │
├──────────────────────────────────────────────────┤
│  [Your question...]          [Send]   [Clear]    │
├──────────────────────────────────────────────────┤
│  📺 Relevant Videos                              │
│  [▶ Thumbnail · Title · 0:42]  [▶ ...]  [▶ ...] │
└──────────────────────────────────────────────────┘
```

Each video card links directly to the relevant timestamp in the source video.

---

## ⚙️ Key Configuration Reference

| Variable | Notebook | Default |
|---|---|---|
| `MAX_WORKERS` | NB00 | `4` (parallel playlist threads) |
| `CHUNK_SIZE` | NB01 | `700` chars |
| `CHUNK_OVERLAP` | NB01 | `150` chars |
| `EMBEDDING_MODEL` | NB03 | `text-embedding-3-small` |
| `BATCH_SIZE` | NB03 | `100` chunks per upsert |
| `CHAT_MODEL` | NB04 | `gpt-4o-mini` |
| `TOP_K` | NB04 | `5` chunks retrieved per query |

> **Switching embedding models?** You must delete the existing Pinecone index and recreate it — indexes are dimension-locked.

---

## ☁️ Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `app.py` and `requirements.txt`
3. Add your API keys under **Settings → Secrets**
4. The Space auto-builds and deploys

---

## 🛠️ Common Issues

| Problem | Fix |
|---|---|
| `yt-dlp` returns no metadata | `pip install -U yt-dlp` |
| Rate-limited by YouTube | Increase `SLEEP_BETWEEN_REQUESTS` to `5` in NB00 |
| `No raw files found` in NB01 | Run NB00 first |
| `PineconeException: INVALID_ARGUMENT` | A metadata field is `None` — NB03 cleans these automatically, re-run |
| `AuthenticationError` | Check `OPENAI_API_KEY` in `.env` |
| Agent returns "I don't know" | Try broader phrasing or increase `TOP_K` in NB04 |
| `MemorySaver` import error | `pip install langgraph` |
| Thumbnails not showing | YouTube CDN may be blocked in your region — links still work |
| `agent_config.json` not found | Run NB04 to export it first |

---

## 📚 Detailed Documentation

Each notebook has its own in-depth README inside `detailed readme files/`:

- `README_00_video_extractor.md`
- `README_01_text_processing.md`
- `README_02_corpus_quality_analysis.md`
- `README_03_pinecone_vector_database.md`
- `README_04_rag_agent_and_evaluation.md`
- `README_05_gradio_deployment.md`

---

## 🔒 Security Note

Never commit your `.env` file — it is already covered by `.gitignore`. Use Hugging Face Secrets or your platform's secrets manager for any public deployment.
