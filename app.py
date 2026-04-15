import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_video_id
from utils import get_transcript
from pipeline import create_vector_store, create_qa_chain

# Load API key

api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

# Page config
st.set_page_config(page_title="YouTube RAG Chatbot")

st.title("🎥 YouTube Video Chatbot")

# -------------------------------
# Session state initialization
# -------------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history

# -------------------------------
# Video Input Section
# -------------------------------
st.markdown("### 📺 Step 1: Enter YouTube Video ID")
st.caption("💡 Paste full YouTube link — no need to extract ID manually")
st.info("🌐 Non-English videos are automatically translated to English")
video_input = st.text_input(
    "YouTube URL or Video ID",
    placeholder="Paste full YouTube link or ID"
)

process_button = st.button("Process Video")

# -------------------------------
# Process Video
# -------------------------------


if process_button:
    if not video_input:
        st.warning("⚠️ Please enter a YouTube URL or ID")
    else:
        try:
            video_id = extract_video_id(video_input)
        except Exception:
            st.error("❌ Invalid YouTube URL or ID")
            st.stop()

        with st.spinner("Processing video..."):
            try:
                text = get_transcript(video_id)
                vector_store = create_vector_store(text)
                qa_chain = create_qa_chain(vector_store)

                st.session_state.qa_chain = qa_chain
                st.session_state.messages = []  # reset chat

                st.success("✅ Video processed! You can now chat below 👇")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -------------------------------
# Chat Section
# -------------------------------
st.markdown("### 💬 Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (replaces Ask button)
query = st.chat_input("Ask something about the video...")

# -------------------------------
# Handle Chat Query
# -------------------------------
if query:
    if not st.session_state.qa_chain:
        st.warning("⚠️ Please process a video first!")
    else:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke(query)

                    st.markdown(response)

                    # Save assistant response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About")
st.sidebar.info(
    "This is a YouTube RAG Chatbot built using LangChain, OpenAI, and FAISS.\n\n"
    "1. Enter a youtube ID or the link\n"
    "2. Process the video\n"
    "3. Chat with the video!"
)