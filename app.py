"""
Mechanical Engineering RAG Chatbot — Gradio Deployment
"""

import os
import json
import gradio as gr
from typing import List, Tuple
from datetime import datetime
from dotenv import load_dotenv

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from pydantic import BaseModel, Field

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

load_dotenv()

config_path = os.path.join(os.path.dirname(__file__), "config", "agent_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

CHAT_MODEL      = config.get("chat_model",      "gpt-4o-mini")
EMBEDDING_MODEL = config.get("embedding_model", "text-embedding-3-small")
INDEX_NAME      = config.get("index_name",      "youtube-rag-mechanical-engineering")
NAMESPACE       = config.get("namespace",       "efficient-engineer-v3")
TOP_K           = config.get("top_k",           5)

if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = config.get("langsmith_project", "youtube-rag-chatbot")
    os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY

# ============================================================================
# 2. LANGCHAIN COMPONENTS
# ============================================================================

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME, embedding=embeddings, namespace=NAMESPACE,
)
print(f"✅ Connected — model: {CHAT_MODEL} | index: {INDEX_NAME}")

# ============================================================================
# 3. TOOLS
# ============================================================================

class SearchTranscriptsInput(BaseModel):
    query: str = Field(description="The search query to find relevant transcript content")

class GetVideoInfoInput(BaseModel):
    video_id: str = Field(description="The YouTube video ID to get information about")

class FindVideosInput(BaseModel):
    topic: str = Field(description="The topic to search for videos about")


def search_transcripts(query: str) -> str:
    try:
        results = vectorstore.similarity_search(query, k=TOP_K)
        if not results:
            return "No relevant information found."
        formatted = []
        for i, doc in enumerate(results, 1):
            m     = doc.metadata
            vid   = m.get("video_id",    m.get("videoId",  "unknown"))
            title = m.get("video_title", m.get("title",    "Unknown Video"))
            t     = m.get("start_time",  m.get("start",    0))
            formatted.append(
                f"[Result {i}]\nVideo: {title}\nTime: {t}s\nContent: {doc.page_content}\n"
                f"Link: https://youtube.com/watch?v={vid}&t={int(t)}s\n"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"Error searching transcripts: {e}"


def get_video_info(video_id: str) -> str:
    try:
        results = vectorstore.similarity_search("", k=1, filter={"video_id": video_id})
        if not results:
            return f"Video {video_id} not found."
        m = results[0].metadata
        return (
            f"Title: {m.get('video_title', m.get('title', 'Unknown'))}\n"
            f"Channel: {m.get('channel', 'Unknown')}\n"
            f"Link: https://youtube.com/watch?v={video_id}"
        )
    except Exception as e:
        return f"Error getting video info: {e}"


def find_videos(topic: str) -> str:
    try:
        results = vectorstore.similarity_search(topic, k=TOP_K * 2)
        if not results:
            return "No videos found."
        seen, videos = set(), []
        for doc in results:
            vid = doc.metadata.get("video_id", doc.metadata.get("videoId"))
            if vid and vid not in seen:
                seen.add(vid)
                videos.append({
                    "title":    doc.metadata.get("video_title", doc.metadata.get("title", "Unknown")),
                    "channel":  doc.metadata.get("channel", "Unknown"),
                    "video_id": vid,
                })
            if len(videos) >= TOP_K:
                break
        return "\n\n".join(
            f"{i}. {v['title']}\n   Channel: {v['channel']}\n"
            f"   Link: https://youtube.com/watch?v={v['video_id']}"
            for i, v in enumerate(videos, 1)
        )
    except Exception as e:
        return f"Error finding videos: {e}"


tools = [
    StructuredTool.from_function(
        func=search_transcripts, name="search_transcripts",
        description=(
            "Search YouTube video transcripts for information. Use this when the user asks "
            "questions about engineering concepts, definitions, or explanations."
        ),
        args_schema=SearchTranscriptsInput,
    ),
    StructuredTool.from_function(
        func=find_videos, name="find_videos",
        description=(
            "Find videos about a specific topic. Use this when the user asks 'which videos', "
            "'show me videos', or wants to browse content about a subject."
        ),
        args_schema=FindVideosInput,
    ),
    StructuredTool.from_function(
        func=get_video_info, name="get_video_info",
        description="Get metadata about a specific video by its ID.",
        args_schema=GetVideoInfoInput,
    ),
]

# ============================================================================
# 4. AGENT
# ============================================================================

system_message = """You are an expert mechanical engineering assistant with access to YouTube video transcripts.

IMPORTANT: Base your answers ONLY on the information found in the transcript search results.
Do not use outside knowledge. If the transcripts don't contain enough information, say so clearly.

Guidelines:
- Always call search_transcripts before answering any question
- Use find_videos when users ask "which videos" or "show me videos"
- Always include YouTube links with timestamps in your responses
- Be concise but thorough in explanations

Current date: {current_date}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent          = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools,
    verbose=False, handle_parsing_errors=True, max_iterations=5,
)
print("✅ Agent ready")

# ============================================================================
# 5. VIDEO HELPERS
# ============================================================================

def get_video_suggestions(query: str, n: int = 4) -> List[dict]:
    """Similarity-search Pinecone, return up to n unique videos with timestamps."""
    try:
        results = vectorstore.similarity_search(query, k=n * 3)
    except Exception:
        return []

    seen, videos = set(), []
    for doc in results:
        m     = doc.metadata
        vid   = m.get("video_id",    m.get("videoId"))
        title = m.get("video_title", m.get("title",  "Unknown"))
        ch    = m.get("channel",     "Unknown")
        t     = m.get("start_time",  m.get("start",  0))
        thumb = m.get("thumbnail",   f"https://img.youtube.com/vi/{vid}/hqdefault.jpg")

        if vid and vid not in seen:
            seen.add(vid)
            videos.append({
                "video_id":      vid,
                "title":         title,
                "channel":       ch,
                "start_time":    int(t),
                "thumbnail_url": thumb,
                "watch_url":     f"https://www.youtube.com/watch?v={vid}&t={int(t)}s",
            })
        if len(videos) >= n:
            break
    return videos


def build_thumbnail_html(videos: List[dict]) -> str:
    """Render clickable video cards as HTML."""
    if not videos:
        return "<p style='color:#94a3b8; font-size:13px; padding:8px 0;'>No related videos found.</p>"

    cards = []
    for v in videos:
        mins  = v["start_time"] // 60
        secs  = v["start_time"] %  60
        ts    = f"{mins}:{secs:02d}"
        title = v["title"][:52] + "…" if len(v["title"]) > 52 else v["title"]
        ch    = v["channel"][:34] + "…" if len(v["channel"]) > 34 else v["channel"]

        cards.append(f"""
        <a href="{v['watch_url']}" target="_blank" rel="noopener" style="
            text-decoration: none;
            display: block;
            width: 200px;
            flex-shrink: 0;
            border-radius: 10px;
            overflow: hidden;
            background: #1e1e3f;
            border: 1px solid #2e2e5e;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5);
        "
        onmouseover="this.style.transform='translateY(-4px)';this.style.boxShadow='0 8px 24px rgba(99,102,241,0.4)'"
        onmouseout="this.style.transform='';this.style.boxShadow='0 2px 10px rgba(0,0,0,0.5)'">
            <div style="position: relative;">
                <img src="{v['thumbnail_url']}"
                     alt="{title}"
                     style="width:100%; height:112px; object-fit:cover; display:block;"
                     onerror="this.src='https://img.youtube.com/vi/{v['video_id']}/hqdefault.jpg'">
                <div style="
                    position: absolute; inset: 0;
                    display: flex; align-items: center; justify-content: center;
                    background: rgba(0,0,0,0.08);
                ">
                    <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
                        <circle cx="18" cy="18" r="18" fill="rgba(0,0,0,0.55)"/>
                        <polygon points="14,11 26,18 14,25" fill="white"/>
                    </svg>
                </div>
                <span style="
                    position: absolute; bottom: 5px; right: 5px;
                    background: rgba(0,0,0,0.85);
                    color: #ffffff;
                    font-size: 10px; font-weight: 700;
                    padding: 2px 5px; border-radius: 3px;
                    font-family: monospace;
                ">▶ {ts}</span>
            </div>
            <div style="padding: 9px 11px 11px; background: #1e1e3f;">
                <div style="
                    font-size: 12px; font-weight: 600;
                    color: #e2e8f0;
                    line-height: 1.4; margin-bottom: 4px;
                ">{title}</div>
                <div style="
                    font-size: 11px; font-weight: 500;
                    color: #818cf8;
                ">{ch}</div>
            </div>
        </a>""")

    inner = "\n".join(cards)
    return f"""
    <div style="padding: 2px 0 4px;">
        <div style="
            font-size: 10px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 1.2px;
            color: #818cf8; margin-bottom: 12px;
        ">📺 Related Videos — click to jump to that exact moment</div>
        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
            {inner}
        </div>
    </div>
    """

# ============================================================================
# 6. CHAT + VIDEO FUNCTIONS
# ============================================================================

_session_histories: dict = {}
_last_queries: dict = {}   # stores query per session for "Learn more" button


def chat_with_agent(
    message: str,
    history: List[dict],
    request: gr.Request,
) -> Tuple[str, List[dict]]:
    """Run the agent. Returns (cleared_input, updated_history). No thumbnails yet."""
    session_key = getattr(request, "session_hash", "default")

    if session_key not in _session_histories:
        _session_histories[session_key] = []
    session_history = _session_histories[session_key]

    try:
        response = agent_executor.invoke({
            "input":        message,
            "chat_history": session_history,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        })
        answer = response.get("output", "Sorry, I couldn't generate a response.")
    except Exception as e:
        answer = f"Error: {e}"

    session_history.append(HumanMessage(content=message))
    session_history.append(AIMessage(content=answer))

    # Save query so "Learn more" can use it without re-asking the user
    _last_queries[session_key] = message + " " + answer[:200]

    return "", history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": answer},
    ]


def show_videos(history: List[dict], request: gr.Request) -> str:
    """Fetch and render thumbnails for the most recent question."""
    if not history:
        return "<p style='color:#94a3b8; font-size:13px;'>Ask a question first!</p>"

    session_key = getattr(request, "session_hash", "default")
    # Find last user message from history
    last_user_msg = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    query = _last_queries.get(session_key) or last_user_msg

    videos = get_video_suggestions(query, n=4)
    return build_thumbnail_html(videos)

# ============================================================================
# 7. UI
# ============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Space+Mono:wght@400;700&display=swap');

/* ═══════════════════════════════════════════════════════════
   DARK THEME  (default — Gradio dark mode or System=dark)
   ═══════════════════════════════════════════════════════════ */
:root,
.dark {
    --bg:      #0f0f1e;
    --surf:    #161628;
    --surf2:   #1e1e3a;
    --border:  #2a2a4e;
    --accent:  #818cf8;
    --accent2: #a5b4fc;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --green:   #34d399;
    --btn-txt: #c4c4e8;
    --btn-bg:  rgba(255,255,255,0.06);
    --btn-bdr: #3a3a6a;
    --r:       12px;
}

/* ═══════════════════════════════════════════════════════════
   LIGHT THEME  (Gradio light mode or System=light)
   ═══════════════════════════════════════════════════════════ */
.light,
body:not(.dark) {
    --bg:      #f0f2ff;
    --surf:    #ffffff;
    --surf2:   #eef0ff;
    --border:  #c7c9e8;
    --accent:  #4f46e5;
    --accent2: #4338ca;
    --text:    #1e1b4b;
    --muted:   #4b5563;
    --green:   #059669;
    --btn-txt: #3730a3;
    --btn-bg:  rgba(79,70,229,0.08);
    --btn-bdr: #a5b4fc;
    --r:       12px;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

body,
.gradio-container,
.gradio-container > div {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container,
.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container div {
    color: var(--text) !important;
}

/* ── Header ── */
.app-header { text-align: center; padding: 26px 0 14px; }
.app-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent2) !important;
    letter-spacing: -.5px;
    margin: 0 0 5px;
}
.app-header h1 .rag-word { color: var(--accent) !important; }
.app-header p { color: var(--muted) !important; font-size: .87rem; margin: 0; }
.status-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(99,102,241,.1); border: 1px solid rgba(99,102,241,.3);
    color: var(--accent2) !important;
    padding: 4px 14px; border-radius: 99px;
    font-size: 10px; font-weight: 700; letter-spacing: .9px;
    text-transform: uppercase; margin-top: 11px;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); box-shadow: 0 0 6px var(--green);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Chatbot container ── */
#chatbot {
    background-color: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
}
/* user bubble — always indigo gradient, white text */
#chatbot .message.user {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    color: #ffffff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 11px 15px !important;
    font-size: .9rem !important;
}
#chatbot .message.user * { color: #ffffff !important; }
/* bot bubble */
#chatbot .message.bot {
    background-color: var(--surf2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 13px 17px !important;
    font-size: .9rem !important;
    line-height: 1.7 !important;
}
#chatbot .message.bot p,
#chatbot .message.bot li,
#chatbot .message.bot span { color: var(--text) !important; }
#chatbot .message.bot a    { color: var(--accent) !important; }
#chatbot .message.bot strong { color: var(--accent2) !important; font-weight: 700 !important; }
#chatbot .message.bot h1,
#chatbot .message.bot h2,
#chatbot .message.bot h3,
#chatbot .message.bot h4,
#chatbot .message.bot h5,
#chatbot .message.bot h6 {
    color: var(--accent2) !important;
    font-weight: 700 !important;
    margin: 10px 0 4px !important;
}
#chatbot .message.bot code {
    background: rgba(99,102,241,.12) !important;
    color: var(--accent) !important;
    padding: 1px 5px; border-radius: 4px;
}
#chatbot .message.bot pre {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
}
#chatbot .message.bot pre code {
    background: transparent !important;
    color: var(--accent) !important;
}
#chatbot .message.pending {
    background-color: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}
#chatbot .message.pending span,
#chatbot .message.pending p { color: var(--muted) !important; }

/* ── Text input ── */
#msg-input textarea {
    background-color: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .92rem !important;
    padding: 13px 15px !important;
    resize: none !important;
    transition: border-color .2s;
}
#msg-input textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
    outline: none !important;
}
#msg-input textarea::placeholder { color: var(--muted) !important; }

/* ── Send button ── */
#send-btn {
    background: linear-gradient(135deg, #4338ca, #7c3aed) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: .88rem !important;
    cursor: pointer !important;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 4px 14px rgba(99,102,241,.4) !important;
    min-height: 50px !important;
}
#send-btn:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
/* ensure Send label is always white */
#send-btn span,
#send-btn * { color: #ffffff !important; }

/* ── Learn more button ── */
#learn-btn button {
    background: rgba(99,102,241,.1) !important;
    border: 1px solid rgba(99,102,241,.4) !important;
    border-radius: 10px !important;
    color: var(--accent2) !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: background .2s, border-color .2s, transform .15s !important;
}
#learn-btn button:hover {
    background: rgba(99,102,241,.22) !important;
    border-color: var(--accent) !important;
    transform: translateY(-1px) !important;
}
#learn-btn button * { color: var(--accent2) !important; }

/* ── Action buttons (🗑 Clear / ↺ Retry / ✕ Hide) ── */
/* FIX: was #555580 (nearly invisible on dark). Now uses --btn-txt which is
   light on dark theme and dark-indigo on light theme — always readable. */
.action-btn button {
    background: var(--btn-bg) !important;
    border: 1px solid var(--btn-bdr) !important;
    color: var(--btn-txt) !important;
    border-radius: 8px !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
    cursor: pointer !important;
    transition: border-color .2s, color .2s, background .2s !important;
}
.action-btn button:hover {
    background: rgba(99,102,241,.15) !important;
    border-color: var(--accent) !important;
    color: var(--accent2) !important;
}
/* guarantee the emoji/text inside is never white-on-white or dark-on-dark */
.action-btn button span,
.action-btn button * {
    color: inherit !important;
}

/* ── Thumbnail panel ── */
#thumb-panel {
    background-color: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 14px !important;
}

/* ── Example chips ── */
.example-chip button {
    background: var(--btn-bg) !important;
    border: 1px solid var(--btn-bdr) !important;
    color: var(--accent2) !important;
    border-radius: 99px !important;
    padding: 5px 14px !important;
    font-size: 11.5px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background .18s !important;
    white-space: nowrap !important;
}
.example-chip button:hover { background: rgba(99,102,241,.18) !important; }
.example-chip button * { color: var(--accent2) !important; }

/* ── Section label ── */
.section-label {
    font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.1px;
    color: var(--muted) !important; margin-bottom: 6px;
}

/* ── Processing / loading badges ── */
.progress-text,
footer .progress-text,
.toast-wrap,
[class*="progress-text"],
[class*="toast"],
.svelte-1yrwa15 {
    background: var(--surf2) !important;
    color: var(--accent2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}
[class*="progress-bar-wrap"] {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
#msg-input .wrap,
#msg-input [class*="wrap"],
#msg-input [class*="eta"],
#msg-input [class*="status"],
#msg-input [class*="timer"],
#msg-input [class*="loading"],
.generating,
[class*="eta-bar"],
[class*="timer"] {
    background: var(--surf2) !important;
    color: var(--accent2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    padding: 2px 8px !important;
}
#msg-input [class*="eta"] *,
#msg-input [class*="status"] *,
.generating * { color: var(--accent2) !important; }

/* ── Animated dots ── */
.dot-flashing,
.dot-flashing::before,
.dot-flashing::after { background-color: var(--accent2) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
"""

EXAMPLES = [
    "What is the coefficient of friction?",
    "Explain Young's modulus",
    "What is fatigue failure in metals?",
    "How does tensile strength differ from yield strength?",
    "Which videos cover stress and strain?",
]


def build_ui():
    with gr.Blocks(css=CSS, title="Engineering RAG Assistant") as demo:

        # ── Header ───────────────────────────────────────────────────────────
        gr.HTML(f"""
        <style>
        .progress-text {{
            background: var(--surf2) !important;
            color: var(--accent2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-size: 11px !important;
            font-weight: 600 !important;
            padding: 2px 10px !important;
        }}
        .progress-text * {{ color: var(--accent2) !important; }}
        [class*="progress-bar-wrap"],
        [class*="progressbar"] {{
            background: var(--surf) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }}
        </style>
        <div class="app-header">
            <h1>⚙️ Engineering <span>RAG</span> Assistant</h1>
            <p>Answers pulled directly from YouTube engineering transcripts</p>
            <div class="status-badge">
                <div class="status-dot"></div>
                {CHAT_MODEL} &nbsp;·&nbsp; {INDEX_NAME}
            </div>
        </div>
        """)

        # Force dark theme as default on page load
        gr.HTML("""
        <script>
        (function() {
            var applyDark = function() {
                document.body.classList.add('dark');
                document.body.classList.remove('light');
                var root = document.querySelector('gradio-app');
                if (root) {
                    root.classList.add('dark');
                    root.classList.remove('light');
                }
            };
            applyDark();
            // Re-apply after Gradio hydrates
            setTimeout(applyDark, 100);
            setTimeout(applyDark, 500);
        })();
        </script>
        """)
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            height=460,
            show_label=False,
            render_markdown=True,
            type="messages",
        )

        # ── Input ─────────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            msg_input = gr.Textbox(
                placeholder="Ask a mechanical engineering question…",
                show_label=False,
                elem_id="msg-input",
                scale=9,
                lines=1,
                max_lines=4,
                autofocus=True,
            )
            send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1, min_width=90)

        # ── Utility buttons ───────────────────────────────────────────────────
        with gr.Row():
            clear_btn = gr.Button("🗑 Clear",  elem_classes="action-btn", scale=1)
            retry_btn = gr.Button("↺ Retry",   elem_classes="action-btn", scale=1)
            gr.HTML("<div style='flex:1'></div>")

        # ── Learn more button + hide button ───────────────────────────────────
        with gr.Row():
            learn_btn = gr.Button(
                "📺  Learn more with videos",
                elem_id="learn-btn",
                scale=1,
            )
            close_btn = gr.Button(
                "✕  Hide videos",
                elem_classes="action-btn",
                scale=1,
                visible=False,
            )

        # ── Thumbnail panel (hidden by default) ───────────────────────────────
        thumb_panel = gr.HTML(value="", elem_id="thumb-panel", visible=False)

        # ── Example questions ─────────────────────────────────────────────────
        gr.HTML('<div class="section-label" style="margin-top:18px;">Try asking</div>')
        with gr.Row():
            for ex in EXAMPLES:
                gr.Button(ex, elem_classes="example-chip", size="sm").click(
                    fn=lambda x=ex: x,
                    outputs=msg_input,
                )

        # =====================================================================
        # EVENT WIRING
        # =====================================================================

        def respond(message, chat_history, request: gr.Request):
            if not message.strip():
                return "", chat_history
            return chat_with_agent(message, chat_history, request)

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        # "Learn more" → fetch videos, show panel, swap buttons
        def on_learn_more(history, request: gr.Request):
            html = show_videos(history, request)
            return (
                gr.update(value=html, visible=True),  # thumb_panel: show
                gr.update(visible=False),             # learn_btn: hide
                gr.update(visible=True),              # close_btn: show
            )

        learn_btn.click(
            fn=on_learn_more,
            inputs=[chatbot],
            outputs=[thumb_panel, learn_btn, close_btn],
        )

        # "Hide videos" → hide panel, swap buttons back
        def on_hide():
            return (
                gr.update(value="", visible=False),   # thumb_panel: hide
                gr.update(visible=True),              # learn_btn: show
                gr.update(visible=False),             # close_btn: hide
            )

        close_btn.click(
            fn=on_hide,
            outputs=[thumb_panel, learn_btn, close_btn],
        )

        # Clear: reset everything
        def on_clear():
            return (
                [],                                   # chatbot
                gr.update(value="", visible=False),   # thumb_panel
                gr.update(visible=True),              # learn_btn
                gr.update(visible=False),             # close_btn
            )

        clear_btn.click(
            fn=on_clear,
            outputs=[chatbot, thumb_panel, learn_btn, close_btn],
        )

        # Retry: re-run last question
        def on_retry(history, request: gr.Request):
            if not history:
                return history
            # Find last user message
            last_q = ""
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_q = msg.get("content", "")
                    break
            if not last_q:
                return history
            # Remove the last exchange (last user + last assistant message)
            trimmed = history[:-2] if len(history) >= 2 else []
            _, updated = chat_with_agent(last_q, trimmed, request)
            return updated

        retry_btn.click(
            fn=on_retry,
            inputs=[chatbot],
            outputs=[chatbot],
        )

    return demo


# ============================================================================
# 8. LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Engineering RAG Assistant")
    print(f"   Model  : {CHAT_MODEL}")
    print(f"   Index  : {INDEX_NAME} / {NAMESPACE}")
    print(f"   Tools  : {', '.join(t.name for t in tools)}")
    print("=" * 70 + "\n")

    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
