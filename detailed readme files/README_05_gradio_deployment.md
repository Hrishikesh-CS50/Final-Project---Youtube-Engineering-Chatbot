# Notebook 05 — Gradio Deployment

## Purpose

Tests the complete Gradio chat interface **inside the notebook** before launching the standalone `app.py`. This notebook mirrors `app.py` exactly — use it to verify that the agent, thumbnail generation, and chat UI all work correctly end-to-end.

## Pipeline Position

```
00_video_extractor ► 01_text_processing ► 02_quality_analysis ► 03_pinecone_upload ► 04_agent_evaluation ► [05_gradio_deployment]
                                                                                                                        ↓
                                                                                                                    app.py
```

---

## What It Does

1. Installs / verifies Gradio (`>=4.44.0`)
2. Loads `agent_config.json` produced by Notebook 04
3. Re-initialises all LangChain components (LLM, embeddings, Pinecone VectorStore)
4. Re-defines the same three tools as NB04 (`search_transcripts`, `get_video_info`, `find_videos`)
5. Rebuilds the `AgentExecutor` with identical settings
6. Defines two new UI helper functions:
   - `get_video_suggestions(query, n)` — runs a Pinecone similarity search and returns unique videos with thumbnail URLs and exact timestamps
   - `build_thumbnail_html(videos)` — renders the results as a horizontal strip of clickable HTML cards
7. Tests thumbnail generation inline in the notebook
8. Defines `chat_with_agent()` — the function Gradio calls on every user message
9. Runs a single end-to-end test question
10. Launches the full Gradio UI inside the notebook

---

## Prerequisites

### Python Packages

```bash
pip install gradio>=4.44.0 langchain langchain-openai langchain-pinecone python-dotenv pydantic
```

### API Keys

Same `.env` as Notebook 04:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=ls__...
PINECONE_INDEX_NAME=youtube-rag-mechanical-engineering
PINECONE_NAMESPACE=efficient-engineer-v3
```

### Input

- `agent_config.json` — produced by **Notebook 04**
- A populated Pinecone index — produced by **Notebook 03**

---

## Configuration

This notebook reads all configuration from `agent_config.json`. No hardcoded settings need to be changed here. If you want to override the model or namespace, edit `agent_config.json` or the corresponding `.env` values.

---

## How to Run

1. Ensure Notebooks 03 and 04 have been run successfully
2. Open `05_gradio_deployment.ipynb`
3. Run **Steps 1–6** sequentially to initialise everything
4. Run **Step 5** to verify thumbnail cards render correctly in the notebook
5. Run **Step 7** for a single end-to-end test before launching the UI
6. Run **Step 8** to launch the Gradio interface (a local URL will appear in the output)

---

## Key Functions

### `get_video_suggestions(query, n=4)`

Runs a Pinecone similarity search for `query` and returns up to `n` unique videos. Each result is a dict:

```python
{
    "video_id":      "abc123XYZ89",
    "title":         "Introduction to Friction",
    "channel":       "Efficient Engineer",
    "start_time":    12,                  # seconds
    "thumbnail_url": "https://i.ytimg.com/vi/abc123XYZ89/hqdefault.jpg",
    "watch_url":     "https://youtube.com/watch?v=abc123XYZ89&t=12"
}
```

The `watch_url` includes `&t=` so clicking a card jumps directly to the relevant timestamp.

### `build_thumbnail_html(videos)`

Converts the list from `get_video_suggestions()` into an HTML string of cards. Each card shows:
- YouTube thumbnail image
- Video title
- Channel name
- A timestamp badge (e.g. `0:12`)

The HTML is rendered in the Gradio `HTML` component below the chat window.

### `chat_with_agent(message, history)`

Called by Gradio on every user message. Returns a tuple of `(answer_text, thumbnail_html)`:

1. Invokes the `AgentExecutor` with the question and conversation history
2. Calls `get_video_suggestions()` using `question + first 200 chars of answer` as the search query
3. Returns the agent's answer and the rendered thumbnail strip

---

## Gradio UI Layout

```
┌─────────────────────────────────────────────────────────┐
│  🔧 YouTube Engineering Chatbot                          │
├─────────────────────────────────────────────────────────┤
│  Chat window (Chatbot component)                         │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  [Text input]                    [Send] [Clear]          │
├─────────────────────────────────────────────────────────┤
│  📺 Relevant Videos                                      │
│  [Thumbnail card] [Thumbnail card] [Thumbnail card]      │
└─────────────────────────────────────────────────────────┘
```

---

## Launching `app.py` (Production)

Once you are satisfied with the notebook tests, launch the standalone app:

```bash
cd /path/to/Final-Project---Youtube-Engineering-Chatbot
python app.py
```

`app.py` (34 KB) contains the same logic as this notebook but is structured as a single-file Gradio application. It auto-reads `.env` and `agent_config.json`.

To deploy publicly:
```python
demo.launch(share=True)   # generates a temporary public URL
```

---

## Common Issues

| Problem | Fix |
|---|---|
| `agent_config.json not found` | Run Notebook 04 first to export the config |
| Thumbnails not showing | YouTube may block thumbnail CDN in some regions. The cards will show broken image icons but links still work. |
| Gradio version incompatible | `pip install -U gradio` — requires `>=4.44.0` |
| Chat history lost on page refresh | Expected — session memory is in-process. For persistent history, integrate a database layer. |
| `share=True` tunnel fails | Use a local deployment or deploy to Hugging Face Spaces instead |

---

## Deploying to Hugging Face Spaces

1. Create a new Space (Gradio SDK) at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload `app.py` and `requirements.txt`
3. Add your API keys as **Space Secrets** (Settings → Secrets)
4. The Space will auto-build and deploy

---

## End of Pipeline

This notebook completes the full pipeline:

```
YouTube URLs → Raw JSON → Chunks → Quality Check → Pinecone → RAG Agent → Gradio UI
```
