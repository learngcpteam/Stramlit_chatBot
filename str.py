import streamlit as st
import cohere
from PyPDF2 import PdfReader
import pandas as pd
import pytesseract
from PIL import Image
from io import StringIO

API_KEY = "Q5zKRhI14zbXktk8fx8kwdSYRYmYPHNkZPyVZxCF"
cohere_client = cohere.Client(API_KEY)

# Page Configuration
st.set_page_config(
    page_title="Oracle ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar with Model Settings
st.sidebar.title("🤖 ChatBot Settings")
st.sidebar.markdown("### LLM Model Selection")
llm_model = st.sidebar.selectbox(
    "Choose LLM Model:", ["command-nightly", "command-light", "command-light-nightly"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, step=0.1)
top_p = st.sidebar.slider("Top-P", 0.0, 1.0, 0.75, step=0.05)
top_k = st.sidebar.slider("Top-K", 0, 500, 50, step=10)

st.sidebar.markdown("### Additional Options")
if st.sidebar.button("Reset Chat"):
    st.session_state["chat_history"] = []

# Header Section
st.markdown(
    """
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; text-align: center;">
        <h1 style="color: #333;">Oracle Generative AI Playground 🤖</h1>
        <p style="color: #666;">Upload a document or ask a question to interact with the AI assistant.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# File Upload
uploaded_file = st.file_uploader(
    "Attach a file (optional):",
    type=["pdf", "png", "jpg", "csv", "txt", "docx", "xlsx"],
)
document_content = ""

if uploaded_file:
    try:
        if uploaded_file.type == "application/pdf":
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                document_content += page.extract_text()
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            image = Image.open(uploaded_file)
            document_content = pytesseract.image_to_string(image)
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            document_content = df.to_string(index=False)
        elif uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            document_content = stringio.read()
        else:
            st.error("Unsupported file type.")
    except Exception as e:
        st.error(f"Error processing the file: {e}")

# Chat History Initialization
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display Chat Messages
for message in st.session_state.chat_history:
    role = message["role"]
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Type your message here..."):
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Display spinner while generating a response
    with st.spinner("Thinking..."):
        prompt = f"Document content:\n{document_content}\n\nUser question: {user_input}\nAnswer:" if document_content else f"User question: {user_input}\nAnswer:"
        try:
            response = cohere_client.generate(
                model=llm_model,
                prompt=prompt,
                max_tokens=500,
                temperature=temperature,
                p=top_p,
                k=top_k,
            )
            bot_response = response.generations[0].text
        except Exception as e:
            bot_response = f"Error generating response: {e}"

    # Append bot response
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot_response)
