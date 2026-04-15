import streamlit as st
import os
from dotenv import load_dotenv

from utils import extract_video_id, get_transcript
from pipeline import create_vector_store, create_qa_chain

# -------------------------------
# Load API Key (works locally + cloud)
# -------------------------------
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="YouTube RAG Chatbot", layout="centered")

st.title("🎥 YouTube Chatbot")

# -------------------------------
# Session State
# -------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# Video Input Section
# -------------------------------
st.markdown("### 📺 Add Video")

video_input = st.text_input(
    "Paste YouTube link or ID",
    placeholder="https://youtube.com/watch?v=..."
)

# Optional manual transcript (clean UX)
with st.expander("⚙️ Optional: Paste transcript manually"):
    manual_text = st.text_area(
        "Paste transcript here",
        placeholder="Use only if auto-fetch fails"
    )

process_button = st.button("Process Video")

# -------------------------------
# Process Video
# -------------------------------
if process_button:
    if not video_input and not manual_text:
        st.warning("Please provide a YouTube link or transcript.")
    else:
        with st.spinner("Processing video... ⏳"):
            try:
                if manual_text:
                    text = manual_text
                else:
                    video_id = extract_video_id(video_input)
                    text = get_transcript(video_id)

                vector_store = create_vector_store(text)
                qa_chain = create_qa_chain(vector_store)

                st.session_state.qa_chain = qa_chain
                st.session_state.messages = []

                st.success("✅ Ready! Ask questions below 👇")

            except Exception:
                st.warning("⚠️ Couldn't fetch transcript. Try pasting it manually.")

# -------------------------------
# Chat Section
# -------------------------------
st.markdown("### 💬 Chat with the Video")

# Show hint if not ready
if not st.session_state.qa_chain:
    st.info("👆 Add a video to start chatting")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask something about the video...")

# -------------------------------
# Handle Chat
# -------------------------------
if query:
    if not st.session_state.qa_chain:
        st.warning("Please process a video first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke(query)

                    st.markdown(response)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About")
st.sidebar.info(
    "Chat with any YouTube video using AI.\n\n"
    "1. Paste video link\n"
    "2. Process\n"
    "3. Ask questions"
)