# Notebook 00 — YouTube Video & Transcript Extractor

## Purpose

Downloads transcripts and metadata from any YouTube URL and saves the results as structured JSON files. This is the **first step** in the pipeline — every subsequent notebook depends on the output produced here.

## Pipeline Position

```
[00_video_extractor] ► 01_text_processing ► 02_quality_analysis ► 03_pinecone_upload ► 04_agent_evaluation ► 05_gradio_deployment
```

---

## What It Does

1. Accepts a single YouTube URL (standard video, YouTube Shorts, or a full playlist)
2. Detects the URL type automatically
3. Fetches video metadata (title, channel, duration, views, upload date, tags, etc.) using `yt-dlp`
4. Fetches the transcript using `youtube-transcript-api` with a priority order:
   - Manual (human-written) transcript — preferred
   - Auto-generated transcript — fallback
   - Any available language — last resort
5. Saves one `video_{id}.json` file per video to `../data/raw/`
6. Supports parallel processing for playlists via `ThreadPoolExecutor`
7. Skips already-downloaded videos by default (`overwrite=False`)

---

## Prerequisites

### Python Packages

```bash
pip install yt-dlp youtube-transcript-api pandas tqdm
```

### Environment Variables

Copy `.env.example` to `.env` — no API keys are required for this notebook specifically, but having the file in place avoids import errors from downstream notebooks.

```
# .env — no keys needed for NB00, but create the file
OPENAI_API_KEY=
PINECONE_API_KEY=
LANGSMITH_API_KEY=
```

### Folder Structure

The notebook auto-creates `../data/raw/` if it does not exist. Your project root should look like:

```
Final-Project---Youtube-Engineering-Chatbot/
├── notebooks/
│   └── 00_video_extractor_universal_url.ipynb   ← you are here
├── data/
│   └── raw/                                      ← output goes here
├── .env
└── requirements.txt
```

---

## Configuration

All tuneable settings live in the **Cell 05 — Configuration** block:

| Variable | Default | Description |
|---|---|---|
| `RAW_DATA_DIR` | `../data/raw` | Where JSON files are saved |
| `LANGUAGES` | `("en",)` | Preferred transcript languages, in order |
| `MAX_WORKERS` | `4` | Threads for parallel playlist collection |
| `SLEEP_BETWEEN_REQUESTS` | `3` | Seconds to wait between API calls |

---

## How to Run

1. Open `00_video_extractor_universal_url.ipynb` in Jupyter
2. Run cells 1–4 (installs, imports, config, helpers)
3. In **Cell 05 — Run Collection**, replace the example URL with your own:

```python
# Single video
youtube_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Full playlist
youtube_url = "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

# YouTube Short
youtube_url = "https://www.youtube.com/shorts/YOUR_SHORT_ID"
```

4. Optionally set `max_videos = 5` to test with a small sample before running the full playlist
5. Run the cell — progress is shown via `tqdm`

---

## Output Format

Each saved file is `../data/raw/video_{video_id}.json` with this structure:

```json
{
  "video_id": "abc123XYZ89",
  "metadata": {
    "title": "Introduction to Friction",
    "url": "https://www.youtube.com/watch?v=abc123XYZ89",
    "duration": 743,
    "channel": "Efficient Engineer",
    "upload_date": "20230415",
    "view_count": 120000,
    "tags": ["engineering", "friction", "tribology"]
  },
  "transcript": [
    { "text": "Welcome to this video on friction.", "start": 0.0, "duration": 3.2 },
    ...
  ],
  "transcript_language": "en",
  "transcript_status": "manual",
  "available_transcripts": [...],
  "collected_at": "2026-03-13T10:00:00"
}
```

`transcript_status` will be one of: `manual`, `auto-generated`, `fallback_any_language`, or `not_available`.

---

## Key Classes & Functions

| Name | Description |
|---|---|
| `detect_url_type(url)` | Returns `"video"`, `"short"`, `"playlist"`, or `"unknown"` |
| `get_video_id(url)` | Extracts the 11-character video ID from any YouTube URL format |
| `YouTubeCollector` | Main class — wraps yt-dlp and youtube-transcript-api |
| `.collect_single_video(url)` | Collects one video |
| `.collect_playlist(url)` | Collects all videos in a playlist (parallel or sequential) |
| `.collect_from_url(url)` | Universal entry point — detects type and dispatches automatically |

---

## Common Issues

| Problem | Fix |
|---|---|
| `TranscriptsDisabled` error | The video has no transcript. It will be saved with `transcript_status: not_available` and an empty transcript list. |
| Rate-limited by YouTube | Increase `SLEEP_BETWEEN_REQUESTS` to `5` or more |
| `yt-dlp` returns no metadata | Update yt-dlp: `pip install -U yt-dlp` |
| Playlist only partially collected | Set `max_videos=None` and `overwrite=False` to resume |

---

## Next Step

Once all JSON files are saved in `../data/raw/`, proceed to:

**`01_text_processing_all_videos_timestamped.ipynb`** — chunks the transcripts into RAG-ready segments with timestamps.
