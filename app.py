"""
Gradio Deployment for YouTube RAG Agent
Loads configuration from agent_config.json and deploys chatbot interface
"""

import os
import json
import gradio as gr
from typing import List, Tuple
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from pydantic import BaseModel, Field

# ============================================================================
# 1. LOAD CONFIGURATION
# ============================================================================

# Load environment variables
load_dotenv()

# Load agent configuration
with open("config/agent_config.json", "r") as f:
    config = json.load(f)

print("✅ Loaded agent configuration:")
print(json.dumps(config, indent=2))

# ============================================================================
# 2. INITIALIZE COMPONENTS
# ============================================================================

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# Model Configuration (from config.json)
CHAT_MODEL = config.get("chat_model", "gpt-4o-mini")
EMBEDDING_MODEL = config.get("embedding_model", "text-embedding-3-small")
INDEX_NAME = config.get("index_name", "youtube-rag-mechanical-engineering")
NAMESPACE = config.get("namespace", "efficient-engineer-v3")
TOP_K = config.get("top_k", 3)

# LangSmith Configuration (optional)
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = config.get("langsmith_project", "youtube-rag-chatbot")
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# Initialize LLM
llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# Initialize Embeddings
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
)

# Initialize Pinecone VectorStore
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)

print(f"✅ Initialized with:")
print(f"   - Model: {CHAT_MODEL}")
print(f"   - Embedding: {EMBEDDING_MODEL}")
print(f"   - Index: {INDEX_NAME}")
print(f"   - Namespace: {NAMESPACE}")

# ============================================================================
# 3. DEFINE TOOL SCHEMAS
# ============================================================================

class SearchTranscriptsInput(BaseModel):
    query: str = Field(description="The search query to find relevant transcript content")

class GetVideoInfoInput(BaseModel):
    video_id: str = Field(description="The YouTube video ID to get information about")

class FindVideosInput(BaseModel):
    topic: str = Field(description="The topic to search for videos about")

# ============================================================================
# 4. DEFINE TOOL FUNCTIONS
# ============================================================================

def search_transcripts(query: str) -> str:
    """Search video transcripts for relevant content."""
    try:
        results = vectorstore.similarity_search(query, k=TOP_K)
        
        if not results:
            return "No relevant information found."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            # Handle different metadata key formats
            video_id = metadata.get('video_id', metadata.get('videoId', 'unknown'))
            title = metadata.get('video_title', metadata.get('title', 'Unknown Video'))
            start_time = metadata.get('start_time', metadata.get('start', 0))
            channel = metadata.get('channel', 'Unknown Channel')
            
            formatted_results.append(
                f"**[Source {i}]**\n"
                f"📹 **Video:** {title}\n"
                f"👤 **Channel:** {channel}\n"
                f"⏱️ **Time:** {start_time}s\n"
                f"📝 **Content:** {content}\n"
                f"🔗 **Link:** https://youtube.com/watch?v={video_id}&t={int(start_time)}s\n"
            )
        
        return "\n".join(formatted_results)
    
    except Exception as e:
        return f"Error searching transcripts: {str(e)}"


def get_video_info(video_id: str) -> str:
    """Get metadata about a specific video."""
    try:
        # Search for any chunk from this video
        results = vectorstore.similarity_search(
            query="",
            k=1,
            filter={"video_id": video_id}
        )
        
        if not results:
            return f"Video {video_id} not found in database."
        
        metadata = results[0].metadata
        
        info = (
            f"📹 **Title:** {metadata.get('video_title', metadata.get('title', 'Unknown'))}\n"
            f"👤 **Channel:** {metadata.get('channel', 'Unknown')}\n"
            f"🆔 **Video ID:** {video_id}\n"
            f"🔗 **Link:** https://youtube.com/watch?v={video_id}"
        )
        
        return info
    
    except Exception as e:
        return f"Error getting video info: {str(e)}"


def find_videos(topic: str) -> str:
    """Find videos related to a specific topic."""
    try:
        results = vectorstore.similarity_search(topic, k=TOP_K * 2)
        
        if not results:
            return "No videos found on this topic."
        
        # Extract unique videos
        seen_videos = set()
        videos = []
        
        for doc in results:
            video_id = doc.metadata.get('video_id', doc.metadata.get('videoId'))
            if video_id and video_id not in seen_videos:
                seen_videos.add(video_id)
                videos.append({
                    'title': doc.metadata.get('video_title', doc.metadata.get('title', 'Unknown')),
                    'channel': doc.metadata.get('channel', 'Unknown'),
                    'video_id': video_id
                })
                
            if len(videos) >= TOP_K:
                break
        
        formatted_videos = []
        for i, video in enumerate(videos, 1):
            formatted_videos.append(
                f"**{i}. {video['title']}**\n"
                f"   👤 Channel: {video['channel']}\n"
                f"   🔗 Link: https://youtube.com/watch?v={video['video_id']}"
            )
        
        return "\n\n".join(formatted_videos)
    
    except Exception as e:
        return f"Error finding videos: {str(e)}"

# ============================================================================
# 5. CREATE LANGCHAIN TOOLS
# ============================================================================

tools = [
    StructuredTool.from_function(
        func=search_transcripts,
        name="search_transcripts",
        description=(
            "Search YouTube video transcripts for information. Use this when the user asks "
            "questions about engineering concepts, definitions, or explanations. Returns "
            "relevant transcript segments with timestamps and video links."
        ),
        args_schema=SearchTranscriptsInput,
    ),
    StructuredTool.from_function(
        func=find_videos,
        name="find_videos",
        description=(
            "Find videos about a specific topic. Use this when the user asks 'which videos', "
            "'show me videos', or wants to browse content about a subject. Returns a list of "
            "relevant videos with titles and links."
        ),
        args_schema=FindVideosInput,
    ),
    StructuredTool.from_function(
        func=get_video_info,
        name="get_video_info",
        description=(
            "Get detailed metadata about a specific video by its ID. Use this when you have "
            "a video_id and need more information about that video. Returns title, channel, "
            "and link."
        ),
        args_schema=GetVideoInfoInput,
    ),
]

print(f"✅ Created {len(tools)} agent tools")

# ============================================================================
# 6. CREATE AGENT PROMPT
# ============================================================================

system_message = """You are an expert mechanical engineering assistant with access to YouTube video transcripts from engineering channels.

Your role:
- Answer engineering questions using the transcript search tool
- Provide accurate, detailed technical explanations
- Always cite sources with timestamps when available
- Suggest relevant videos when appropriate

Guidelines:
- Use search_transcripts for concept explanations and definitions
- Use find_videos when users ask "which videos" or "show me videos"
- Always include YouTube links with timestamps in your responses
- Be concise but thorough in explanations
- If information isn't found, say so clearly and suggest related topics

Current date: {current_date}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

print("✅ Agent prompt created")

# ============================================================================
# 7. CREATE AGENT
# ============================================================================

# Create agent
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)

print("✅ Agent executor created")

# ============================================================================
# 8. GRADIO CHAT FUNCTION
# ============================================================================

def chat_with_agent(message: str, history: List[Tuple[str, str]]) -> str:
    """
    Chat function for Gradio interface.
    
    Args:
        message: User's current message
        history: List of (user_msg, bot_msg) tuples from Gradio
    
    Returns:
        Agent's response
    """
    try:
        # Convert Gradio history to LangChain message format
        session_history = []
        for user_msg, bot_msg in history:
            session_history.append(HumanMessage(content=user_msg))
            session_history.append(AIMessage(content=bot_msg))
        
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Invoke agent with history
        response = agent_executor.invoke({
            "input": message,
            "chat_history": session_history,
            "current_date": current_date,
        })
        
        return response.get("output", "I apologize, but I couldn't generate a response.")
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        return f"I apologize, but I encountered an error: {str(e)}"

# ============================================================================
# 9. CREATE GRADIO INTERFACE
# ============================================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 900px !important;
}
"""

# Example questions
examples = [
    ["What is stress in engineering?"],
    ["Explain Young's modulus"],
    ["How do materials fail under fatigue?"],
    ["Which videos discuss carbon fiber?"],
    ["What is the difference between stress and strain?"],
]

# Create Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="🔧 Mechanical Engineering YouTube RAG Assistant",
    description=(
        f"Ask questions about mechanical engineering concepts! I have access to transcripts from "
        f"engineering YouTube channels and can provide detailed answers with video citations.\n\n"
        f"**Model:** {CHAT_MODEL} | **Database:** {INDEX_NAME}"
    ),
    examples=examples,
    theme=gr.themes.Soft(),
    css=custom_css,
    retry_btn="🔄 Retry",
    undo_btn="↩️ Undo",
    clear_btn="🗑️ Clear",
    submit_btn="Send 📨",
)

# ============================================================================
# 10. LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 LAUNCHING GRADIO INTERFACE")
    print("="*80)
    print(f"Model: {CHAT_MODEL}")
    print(f"Index: {INDEX_NAME}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Tools: {', '.join([t.name for t in tools])}")
    print("="*80 + "\n")
    
    # Launch with share=True to get public link
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True for public link
        show_error=True,
    )
