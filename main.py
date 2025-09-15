import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import asyncio
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
import time

# ✅ Load API key securely from Streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is not set. Please add it in Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = gemini_api_key

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
3. Never invent facts.
4. Keep your tone friendly, clear, and concise.
"""

# Async loop setup
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# --- Session State Initialization ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_video_id" not in st.session_state:
        st.session_state.current_video_id = None

# --- Page Layout ---
def configure_page():
    st.set_page_config(page_title="YouTube RAG Chat", page_icon="🎥", layout="centered")
    st.title("⚡ YouTube x RAG Assistant")
    st.markdown("### Transform any YouTube video into an interactive conversation")

# --- Sidebar (Model Selector + Controls) ---
def handle_sidebar():
    st.sidebar.header("⚙️ Configuration")
    selected_model = st.sidebar.selectbox(
        "Generation Models",
        ["gemini-2.5-pro","gemini-2.5-flash","gemini-2.5-flash-lite","gemini-1.5-pro","gemini-1.5-flash"],
        index=0,
    )
    st.session_state.model = selected_model

    st.sidebar.divider()
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

    st.sidebar.divider()
    message_count = len(st.session_state.messages) - 1
    video_processed = "retriever" in st.session_state and st.session_state.get("retriever") is not None
    st.sidebar.metric("Messages", message_count)
    st.sidebar.metric("Video", "✅" if video_processed else "❌")
    return selected_model

def extract_video_id(url_or_id: str) -> str:
    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id if len(url_or_id) == 11 else ""

def display_video_info(video_id: str):
    if video_id:
        st.image(f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg", width=600)
        st.success(f"📺 Video ID: `{video_id}` ready!")

def handle_video_processing(video_id=""):
    if st.button("🚀 Process Video", type="primary"):
        if not video_id:
            st.error("❌ Please enter a valid YouTube URL!")
            return
        with st.spinner("Processing video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("🔄 Extracting transcript...")
                progress_bar.progress(25)
                transcript_list = YouTubeTranscriptApi().get_transcript(video_id)
                transcript = " ".join(item["text"] for item in transcript_list)
            except TranscriptsDisabled:
                st.error("❌ Transcripts are disabled for this video.")
                st.stop()
            except Exception:
                st.error("❌ Failed to fetch transcript.")
                st.stop()

            status_text.text("📄 Splitting into chunks...")
            progress_bar.progress(50)
            chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).create_documents([transcript])

            status_text.text("🧠 Creating embeddings...")
            progress_bar.progress(75)
            embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

            status_text.text("🗂️ Building search index...")
            progress_bar.progress(100)
            retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})
            st.session_state["retriever"] = retriever
            st.success("✅ Video processed! Ready for questions.")
            time.sleep(1.5)
            st.rerun()

def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource()
def get_chat_model(model_name: str): return ChatGoogleGenerativeAI(model=model_name)

def display_chat_messages():
    for msg in st.session_state.messages[1:]:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.write(msg.content)

def handle_user_input(chat_model):
    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"): st.write(prompt)
        retriever = st.session_state.get("retriever")
        if not retriever:
            with st.chat_message("assistant"):
                st.error("❌ Please process a video first.")
            return
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing video..."):
                retrieved_docs = retriever.invoke(prompt)
                if not retrieved_docs:
                    st.warning("🤷 No relevant info found.")
                    return
                parallel_chain = RunnableParallel({"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()})
                parser = StrOutputParser()
                main_chain = parallel_chain | PromptTemplate(template="Based on this transcript:\n{context}\n\nQ: {question}", input_variables=["context","question"]) | chat_model | parser
                message_placeholder = st.empty()
                full_response = ""
                for chunk in main_chain.stream(prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))
                st.rerun()

# --- App Execution ---
init_session_state()
configure_page()
selected_model = handle_sidebar()
video_url = st.text_input("🔗 YouTube Video URL", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
video_id = extract_video_id(video_url) if video_url else ""
st.session_state.current_video_id = video_id
if video_id: display_video_info(video_id)
handle_video_processing(video_id)
chat_model = get_chat_model(selected_model)
display_chat_messages()
handle_user_input(chat_model)
