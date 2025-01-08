import streamlit as st
import cohere
from PyPDF2 import PdfReader
import pandas as pd
import pytesseract
from PIL import Image
# import io

API_KEY = "*****************************"
cohere_client = cohere.Client(API_KEY)

st.set_page_config(
    page_title="LLM Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("LLM Playground")
st.sidebar.markdown("### Model Settings")
llm_model = st.sidebar.selectbox(
    "Choose your LLM Model:", ["command-nightly", "command-light", "command-light-nightly"]
)
temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.3, step=0.05)
top_p = st.sidebar.slider("Top-P:", 0.0, 1.0, 0.75, step=0.05)
top_k = st.sidebar.slider("Top-K:", 0, 500, 0, step=5)
st.sidebar.markdown("---")

st.sidebar.title("About")
st.sidebar.info("This is a application that supports PDF, CSV, and image files for document-based Q&A...!!")

# File Upload
st.title("🤖 ChatBot Playground with for all Document support")
st.subheader("Upload a document (PDF, Image, or CSV) and ask any questions about it.")

uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "png", "jpg", "csv"])

document_content = ""

if uploaded_file:
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

    st.text_area("Extracted Document Content:", document_content, height=200)

# we have Chat UI..
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# text input for user will give query.
user_input = st.text_input("Type your message here..", "")

# user provides input, generate response
if st.button("Send"):
    if user_input.strip() and document_content:
        st.session_state.chat_history.append(("User", user_input))  # append user input chat his..

        try:
            # document content of the prompt.
            prompt = f"Document content:\n{document_content}\n\nUser question: {user_input}\nAnswer:"
            response = cohere_client.generate(
                model=llm_model,
                prompt=prompt,
                max_tokens=500,
                temperature=temperature,
                p=top_p,
                k=top_k,
            )
            st.session_state.chat_history.append(("LLM", response.generations[0].text))
        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please upload a document and enter a message.")

st.markdown("### Chat History")
for sender, message in st.session_state.chat_history:
    if sender == "User":
        st.markdown(f"**{sender}:** {message}")
    else:
        st.markdown(f"🟢 **{sender}:** {message}")