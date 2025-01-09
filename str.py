
# Streamlit and UI imports
import streamlit as st
import datetime
import pytz
from datetime import timedelta
import yaml
from urllib.parse import urlparse, unquote

# LangChain-related imports (grouped together and organized alphabetically)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.llms import Cohere  # Use the correct class for Cohere
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage  # Added this for clarity

# Avatar and Logo Mapping
AVATAR_MAPPING = {
    "user": st.secrets["user_avatar"],
    "assistant": st.secrets["assisstant_avatar"],
}

logo_image_path = st.secrets["logo"]
st.logo(logo_image_path)

# Cohere API Configuration
COHERE_API_KEY = st.secrets.get("cohere_api_key", "")
if not COHERE_API_KEY:
    st.error("Cohere API key is missing in secrets configuration!")

if st.session_state.get("page", "LLM") != st.session_state.current_page:
    # Clear all session state data
    st.session_state.clear()

    # Store the current page for the next comparison
    st.session_state.current_page = st.session_state.get("page", "LLM")

# Streamlit UI
st.header("Cohere LLM Playground")
st.subheader("Powered by Cohere Generative AI")
st.info(
    "`This chat allows interaction with Cohere's chat endpoint. Use the sidebar to change the LLM model, update parameters, and reset the chat.`"
)

# LangChain Prompt Template
DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
Assistant:"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi! I am Cohere Generative AI! How may I assist you?",
}

# Utility Functions

def new_chat():
    """Reset the chat to its initial state."""
    st.session_state["messages"] = [INIT_MESSAGE]
    st.toast("Chat reset!")

def update_params():
    """Update conversation chain parameters."""
    st.session_state["conv_chain"] = init_conversationchain()
    st.toast("Parameters saved!")

# Sidebar Configuration
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        st.button("Save Changes", on_click=update_params, type="primary", use_container_width=True)
    with col2:
        st.button("Reset Chat", on_click=new_chat, type="primary", use_container_width=True)

    # Parameter tuning options
    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Controls randomness of output. Lower values make output more deterministic.",
    )
    MAX_TOKENS = st.slider(
        "Max Tokens",
        min_value=1,
        max_value=4000,
        value=500,
        step=10,
        help="Maximum number of tokens for the response.",
    )
    PRESENCE_PENALTY = st.slider(
        "Presence Penalty",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Encourages the model to talk about new topics.",
    )
    MEMORY_WINDOW = st.slider(
        "Memory Window",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Number of past interactions to remember.",
    )

# Initialize the ConversationChain
def init_conversationchain():
    """Initialize the conversation chain with Cohere."""
    llm = Cohere(
        model="command",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        presence_penalty=PRESENCE_PENALTY,
    )

    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=MEMORY_WINDOW, ai_prefix="Assistant"),
        prompt=PROMPT,
    )

    return conversation

# Initialize the conversation chain
if "conv_chain" not in st.session_state:  # Check if conv_chain exists in session state
    st.session_state["conv_chain"] = init_conversationchain()
conv_chain = st.session_state["conv_chain"]

# Generate AI Response
def generate_response(conversation, input_text):
    """Generate a response from the AI based on user input."""
    human_message = HumanMessage(content=input_text)
    ai_response = conversation.run(input_text)  # Get the full response directly
    return ai_response

if "messages" not in st.session_state:
    st.session_state.messages = [INIT_MESSAGE]
    conv_chain = init_conversationchain()

# Display Chat Messages
for message in st.session_state.messages:
    avatar = AVATAR_MAPPING.get(message["role"], "o.png")  # Default to "o.png" if not found
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=":material/record_voice_over:"):
        st.markdown(user_input)

    # Display a spinner while waiting for the response
    with st.spinner("Thinking..."):
        full_response = generate_response(conv_chain, user_input)

    # Display assistant's response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant", avatar="o.png"):
        st.markdown(full_response)
