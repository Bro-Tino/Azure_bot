import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4",
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# Prompt template
prompt = PromptTemplate.from_template(
    """You are a helpful assistant.
    Previous conversation:
    {chat_history}
    
    Human: {input_text}
    Assistant:"""
)

# Session memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        memory_key="chat_history",
        return_messages=True,
        input_key="input_text"
    )

if "conversation" not in st.session_state:
    st.session_state.conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=st.session_state.memory,
        verbose=False
    )

# Streamlit UI
st.title("💬 Azure GPT-4 Chatbot")
st.markdown("Ask anything. Your conversation will be summarized as it grows.")

user_input = st.text_input("You:", key="input_text")

if user_input:
    response = st.session_state.conversation.predict(input_text=user_input)
    st.markdown(f"**Assistant:** {response}")

# Show memory summary
with st.expander("🧠 Conversation Summary"):
    st.write(st.session_state.memory.buffer)
