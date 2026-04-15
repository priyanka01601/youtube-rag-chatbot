import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_video_id
from utils import get_transcript
from pipeline import create_vector_store, create_qa_chain

# Load API key

api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
st.warning(
    "⚠️ Transcript fetch may fail due to YouTube restrictions.\n"
    "If it fails, please paste transcript manually below."
)

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
manual_text = st.text_area(
    "📄 Or paste transcript manually (if auto-fetch fails)",
    placeholder="Paste full transcript here..."
)
st.caption("💡 Paste full YouTube link — no need to extract ID manually")
st.info("🌐 Non-English videos are automatically translated to English")
st.info("🌐 If transcript fetching fails (YouTube restrictions), paste transcript manually")
video_input = st.text_input(
    "YouTube URL or Video ID",
    placeholder="Paste full YouTube link or ID"
)

process_button = st.button("Process Video")

# -------------------------------
# Process Video
# -------------------------------


if process_button:
    if not video_input and not manual_text:
        st.warning("⚠️ Please enter a YouTube URL or paste transcript")
    else:
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

            st.success("✅ Ready! You can now chat below 👇")

        except Exception as e:
            st.error(
                "❌ Could not fetch transcript automatically.\n"
                "👉 Please paste transcript manually above."
            )

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