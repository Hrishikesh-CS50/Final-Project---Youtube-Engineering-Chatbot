# Notebook 02 — Corpus Quality Analysis

## Purpose

Validates the entire processed chunk corpus before uploading anything to Pinecone. It runs a battery of quality checks — schema completeness, chunk size sanity, timestamp validity, and duplicate detection — and produces a human-readable readiness verdict plus visual charts.

## Pipeline Position

```
00_video_extractor ► 01_text_processing ► [02_quality_analysis] ► 03_pinecone_upload ► 04_agent_evaluation ► 05_gradio_deployment
```

---

## What It Does

1. Discovers all `chunks_*.json` files in `../data/processed/`
2. Loads every chunk into memory and normalises the records into a flat pandas DataFrame
3. Validates each chunk against configurable thresholds:
   - Character count (min/max)
   - Word count (min/max)
   - Timestamp duration (min/max seconds)
   - Required schema fields (`chunk_id`, `video_id`, `text`, `start_time`, `end_time`)
4. Detects duplicate `chunk_id` values
5. Detects exact duplicate chunk texts (using MD5 hashing)
6. Flags `None` / missing metadata fields that must be cleaned before Pinecone upsert
7. Generates per-video statistics and corpus-level summary tables
8. Plots distribution charts (character count, word count, duration)
9. Prints a **PASS / WARN / FAIL** readiness decision

---

## Prerequisites

### Python Packages

```bash
pip install pandas numpy matplotlib
```

### Input

All files matching `../data/processed/chunks_*.json` — produced by **Notebook 01**.

### Folder Structure

```
Final-Project---Youtube-Engineering-Chatbot/
├── notebooks/
│   └── 02_corpus_quality_analysis.ipynb
├── data/
│   └── processed/    ← input (from NB01)
```

---

## Configuration

All thresholds are in **Section 2 — Configuration**:

| Variable | Default | Description |
|---|---|---|
| `PROCESSED_DIR` | `../data/processed` | Where chunk files are read from |
| `MIN_CHARS` | `150` | Minimum characters for a valid chunk |
| `MAX_CHARS` | `1000` | Maximum characters for a valid chunk |
| `MIN_WORDS` | `25` | Minimum word count |
| `MAX_WORDS` | `180` | Maximum word count |
| `MIN_DURATION_SEC` | `5` | Minimum timestamp span in seconds |
| `MAX_DURATION_SEC` | `120` | Maximum timestamp span in seconds |

Adjust these to match your `CHUNK_SIZE` setting from Notebook 01. If you used a larger chunk size (e.g. 1000 chars), raise `MAX_CHARS` accordingly.

---

## How to Run

1. Ensure Notebook 01 has populated `../data/processed/`
2. Open `02_corpus_quality_analysis.ipynb`
3. Run all cells in order (Sections 1 → 9)
4. Review the printed summary and charts at the end

No API keys or environment variables are needed.

---

## Checks Performed

### Schema Checks
Every chunk must contain: `chunk_id`, `video_id`, `text`, `start_time`, `end_time`.

### Size Checks
Chunks outside the `MIN_CHARS`/`MAX_CHARS` range are flagged as `too_short` or `too_long`. These often indicate splitter edge cases or videos with very sparse transcripts.

### Duration Checks
Chunks where `end_time - start_time` falls outside `MIN_DURATION_SEC`/`MAX_DURATION_SEC` are flagged. Long-duration chunks may indicate timestamp propagation errors.

### Duplicate Chunk IDs
A duplicate `chunk_id` means two chunks share the same identifier — this would cause silent overwrites in Pinecone. Must be zero before uploading.

### Duplicate Texts
Identical chunk text (regardless of ID) wastes embedding quota and inflates retrieval noise.

### None / Missing Metadata
Pinecone metadata values cannot be `None`. These are flagged here and cleaned in Notebook 03.

---

## Output

This notebook writes **no files** by default — all output is displayed inline. The report variables (`issues_df`, `df_chunks`, `file_level_summary`) remain in memory for interactive inspection.

Optional report export lines are commented out in the notebook:

```python
# REPORT_JSON = "quality_report_corpus.json"
# VIDEO_CSV   = "quality_report_videos.csv"
# CHUNK_CSV   = "quality_report_chunks.csv"
# ISSUES_CSV  = "quality_report_issues.csv"
```

Uncomment and re-run to save reports to disk.

---

## Issue Severity Levels

| Level | Meaning |
|---|---|
| `ERROR` | Must be fixed before Pinecone upload (e.g. duplicate chunk IDs) |
| `WARNING` | Should be reviewed (e.g. chunks slightly outside size range) |
| `INFO` | Informational only (e.g. auto-generated transcript) |

---

## Common Issues

| Problem | Fix |
|---|---|
| `No chunk files found` | Run Notebook 01 first |
| Many `too_short` chunks | Lower `MIN_CHUNK_SIZE` in NB01, or accept and filter in NB03 |
| `None` metadata values | Expected — Notebook 03 handles cleaning automatically |
| High duplicate text count | Check if the same video was processed twice in NB00 |

---

## Next Step

Once the corpus passes quality checks (zero `ERROR`-level issues), proceed to:

**`03_pinecone_vector_database_all_videos.ipynb`** — embeds the chunks and upserts them into Pinecone.
