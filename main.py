# app.py
import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import asyncio
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import time

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/embedding-001"
RETRIEVER_K = 4
DEFAULT_SYSTEM_MESSAGE = """
You are YouTube RAG Assistant 📺🤖. 
Your role is to help users understand and explore the content of YouTube videos by using the retrieved transcript context. 

Follow these rules:
1. Always prioritize the transcript/context when answering questions about the video. 
   - Summarize, explain, or extract details only from the retrieved text.
   - If the answer is not present in the transcript, clearly say you don’t know.
   
2. Also maintain awareness of the ongoing chat history (previous user and assistant messages in this session). 
   - If the user asks about their previous messages, use the chat history instead of the transcript.
   - For example, if the user asks “what is my name” and they told you earlier, answer from the chat history.

3. Never invent facts. If the context or chat history does not contain the answer, politely say you don’t know. 

4. Keep your tone friendly, clear, and concise. 
   - Use bullet points or short paragraphs if the answer is long. 
   - Do not repeat the system instructions in your answers.
"""

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None


def configure_page():
    st.set_page_config(
        page_title="YouTube RAG Chat",
        page_icon="🎥",
        layout="centered",
    )

    st.title("⚡YouTube x RAG Assistant")
    st.markdown("### Transform any YouTube video into an interactive conversation")


def center_app():
    st.markdown(
        """
        <style>
        .block-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            max-width: 800px;
            margin: auto;
        }
        #transform-any-you-tube-video-into-an-interactive-conversation,#you-tube-x-rag-assistant{
        text-align:center;
        }
        .stTextInput, .stButton {
            width: 100% !important;
            margin: auto;
        }
        @media (max-width: 768px) {
                .stVerticalBlock{
                align-items:center;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def handle_new_video_button():
    if st.sidebar.button("🔄 New Video", use_container_width=True):
        if "retriever" in st.session_state:
            del st.session_state["retriever"]
        if "current_video_id" in st.session_state:
            st.session_state.current_video_id = None

        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]

        st.success("🔄 Ready for new video!")
        time.sleep(1)
        st.rerun()


def handle_sidebar():
    st.sidebar.header("🔑 Configuration")

    # API key directly from st.secrets
    api_key = st.secrets.get("GOOGLE_API_KEY", "")

    if api_key:
        st.session_state.api_key = api_key
        st.sidebar.success("✅ API key loaded from secrets.toml")
    else:
        st.sidebar.error("❌ No API key found in secrets. Please add it in Streamlit settings.")

    st.sidebar.divider()

    selected_model = st.sidebar.selectbox(
        "Generation Models",
        [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash-image-preview",
            "gemini-live-2.5-flash-preview",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-live-001",
            "gemini-2.0-flash-live-preview-04-09",
            "gemini-2.0-flash-preview-image-generation",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ],
        index=0,
        help="Choose the Gemini model for generation",
    )

    st.session_state.model = selected_model

    st.sidebar.divider()
    st.sidebar.subheader("💬 Chat Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
            st.rerun()

    with col2:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

    handle_new_video_button()

    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Info")

    message_count = len(st.session_state.messages) - 1
    video_processed = (
        "retriever" in st.session_state
        and st.session_state.get("retriever") is not None
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Messages", message_count)
    with col2:
        st.metric("Video", "✅" if video_processed else "❌")

    if video_processed:
        st.sidebar.success("🎥 Video ready for chat")
    else:
        st.sidebar.info("📹 No video processed yet")

    st.sidebar.info(f"**Current Model:**\n{selected_model}")

    if message_count > 0:
        st.sidebar.divider()
        chat_text = ""
        for msg in st.session_state.messages[1:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            chat_text += f"{role}: {msg.content}\n\n"

        st.sidebar.download_button(
            "📥 Download Chat",
            chat_text,
            f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            use_container_width=True,
            help="Download your conversation history",
        )

    video_url = st.text_input(
        "🔗 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID"
    )
    video_id = extract_video_id(video_url) if video_url else ""
    st.session_state.current_video_id = video_id

    if video_id:
        display_video_info(video_id)
    elif video_url and not video_id:
        st.error("❌ Invalid YouTube URL format")
        st.info("💡 Please use: youtube.com/watch?v=... or youtu.be/...")

    return selected_model, video_id, st.session_state.get("api_key")


# --- rest of code remains same --- #

init_session_state()
configure_page()
center_app()
selected_model, video_id, user_api_key = handle_sidebar()
handle_video_processing(video_id)
chat_model = None
if user_api_key:
    chat_model = get_chat_model(selected_model, user_api_key)

display_chat_messages()

if chat_model is None:
    st.warning("Please add your Google Gemini API key to Streamlit secrets.")

handle_user_input(chat_model, input_disabled=(chat_model is None))
