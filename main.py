

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os

from agents import Agent, function_tool, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import asyncio

# Load environment variables

gemini_api_key = st.secrets["GOOGLE_API_KEY"]
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Setup model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Agent instructions
instructions = (
    "You provide help with tasks related to YouTube videos. "
    "Always use the `fetch_youtube_transcript` tool to fetch the transcript of a YouTube video."
)

# Tool function
@function_tool
def fetch_youtube_transcript(url: str) -> str:
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    video_id_match = re.search(video_id_pattern, url)

    if not video_id_match:
        raise ValueError("Invalid YouTube URL")

    video_id = video_id_match.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatted_entries = []
        for entry in transcript:
            minutes = int(entry['start'] // 60)
            seconds = int(entry['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_entries.append(f"{timestamp} {entry['text']}")
        return "\n".join(formatted_entries)
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# Define the agent
agent = Agent(
    name="YouTube Transcript Agent",
    instructions=instructions,
    tools=[fetch_youtube_transcript],
)

# -------------------- Streamlit UI --------------------
st.title("🎬 YouTube Transcript Agent")
st.write("Paste a YouTube video link and ask questions like:\n- 'Summarize this video'\n- 'List key points'\n- 'What is the video about?'")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("Your message:", key="input_box")

async def process_message(message):
    input_items = [{"role": "user", "content": message}]
    response_text = ""

    result = Runner.run_streamed(agent, input_items, run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            response_text += event.data.delta
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                response_text += "\n⏳ Fetching transcript..."
            elif event.item.type == "tool_call_output_item":
                response_text += f"\n✅ Transcript fetched:\n{event.item.output}"

    return response_text

# Handle submit
if st.button("Send") and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Run async agent in Streamlit
    response = asyncio.run(process_message(user_input))
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"**You:** {chat['content']}")
    else:
        st.markdown(f"**Agent:** {chat['content']}")
