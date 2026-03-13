# Notebook 01 — Text Processing: Timestamped Transcript Chunking

## Purpose

Reads all raw `video_*.json` files produced by Notebook 00 and converts the transcripts into small, overlapping **RAG-ready chunks** with accurate timestamps. Each chunk is saved as a processed JSON file ready for vector embedding.

## Pipeline Position

```
00_video_extractor ► [01_text_processing] ► 02_quality_analysis ► 03_pinecone_upload ► 04_agent_evaluation ► 05_gradio_deployment
```

---

## What It Does

1. Discovers all `video_*.json` files in `../data/raw/`
2. Deduplicates the file list to prevent double-processing
3. Assembles the raw transcript segments into a single continuous text per video
4. Splits the text into overlapping chunks using **LangChain's `RecursiveCharacterTextSplitter`**
5. Attaches timestamp metadata to every chunk (`start_time`, `end_time`, `start_time_str`, `end_time_str`, `source_segments`)
6. Skips videos with missing or invalid transcripts gracefully
7. Saves one `chunks_{video_id}.json` file per video to `../data/processed/`
8. Writes a batch-level `processing_summary.json` at the end

---

## Prerequisites

### Python Packages

```bash
pip install langchain langchain-text-splitters langchain-core pandas tqdm
```

### Input

All files matching `../data/raw/video_*.json` — produced by **Notebook 00**.

### Folder Structure

```
Final-Project---Youtube-Engineering-Chatbot/
├── notebooks/
│   └── 01_text_processing_all_videos_timestamped.ipynb
├── data/
│   ├── raw/          ← input (from NB00)
│   └── processed/    ← output (auto-created)
```

---

## Configuration

All settings are in **Section 2 — Configuration**:

| Variable | Default | Description |
|---|---|---|
| `RAW_DIR` | `../data/raw` | Where raw JSON files are read from |
| `PROCESSED_DIR` | `../data/processed` | Where chunk files are written |
| `CHUNK_SIZE` | `700` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap in characters between adjacent chunks |
| `MIN_CHUNK_SIZE` | `200` | Minimum characters for a chunk to be kept |
| `CHUNK_FILE_PREFIX` | `chunks_` | Prefix for output filenames |

> **Tip:** A `CHUNK_SIZE` of 700 with `CHUNK_OVERLAP` of 150 is optimised for OpenAI `text-embedding-3-small`. If you switch embedding models, revisit these values.

---

## How to Run

1. Ensure Notebook 00 has already populated `../data/raw/`
2. Open `01_text_processing_all_videos_timestamped.ipynb`
3. Run all cells in order (Sections 1 → 7)

There is nothing to configure if you are using the default paths. The notebook processes every file it finds automatically.

---

## Output Format

Each file is `../data/processed/chunks_{video_id}.json` — a JSON array of chunk objects:

```json
[
  {
    "chunk_id": "abc123XYZ89_chunk_000",
    "video_id": "abc123XYZ89",
    "video_title": "Introduction to Friction",
    "text": "Friction is a force that resists relative motion between two surfaces...",
    "start_time": 12.4,
    "end_time": 43.1,
    "start_time_str": "00:12",
    "end_time_str": "00:43",
    "source_segments": [0, 1, 2, 3],
    "metadata": {
      "channel": "Efficient Engineer",
      "url": "https://www.youtube.com/watch?v=abc123XYZ89",
      "upload_date": "20230415"
    }
  },
  ...
]
```

### Summary File

`../data/processed/processing_summary.json` records the run:

```json
{
  "processed_at": "2026-03-13T10:00:00",
  "total_videos": 65,
  "total_chunks": 4820,
  "chunk_size": 700,
  "chunk_overlap": 150,
  "text_splitter": "LangChain RecursiveCharacterTextSplitter",
  "videos": [...]
}
```

---

## Key Classes & Functions

| Name | Description |
|---|---|
| `TranscriptProcessor` | Main class wrapping LangChain's text splitter |
| `.process_video(video_data)` | Processes one video dict → list of chunk dicts |
| `format_seconds(seconds)` | Converts float seconds to `MM:SS` or `HH:MM:SS` string |
| `clean_text(text)` | Normalises whitespace and removes control characters |

### LangChain Splitter Settings

```python
RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

The splitter tries to break on paragraph → line → sentence → word boundaries, preserving semantic coherence.

---

## Common Issues

| Problem | Fix |
|---|---|
| `No raw files found` | Run Notebook 00 first to populate `../data/raw/` |
| Video skipped — empty transcript | The raw JSON has `transcript: []`. Check `transcript_status` in the raw file. |
| `None` values in output chunks | Expected for some metadata fields. Notebook 02 flags and Notebook 03 cleans these before upserting to Pinecone. |
| Very few chunks per video | Video may be short or transcript may be sparse. Check `word_count` in the summary. |

---

## Next Step

After all chunk files are in `../data/processed/`, proceed to:

**`02_corpus_quality_analysis.ipynb`** — validates chunk quality across the entire corpus before uploading to the vector database.
