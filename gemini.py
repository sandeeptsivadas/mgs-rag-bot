import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def list_available_models():
    """Lists the available Generative AI models."""
    genai.configure(api_key=st.secrets["google_api_key"])
    models = genai.list_models()
    available_model_names = [model.name for model in models]
    st.session_state.status.write(f":material/format-list-bulleted: Available models: {available_model_names}")
    return available_model_names

def get_llm():
    """Initialize a Gemini LLM from Google if available."""
    st.session_state.status.write(":material/emoji_objects: Setting up Gemini LLM...")
    available_models = list_available_models()
    if "gemini-pro" in available_models:
        model_name = "gemini-pro"
    elif "gemini-1.0-pro" in available_models:
        model_name = "gemini-1.0-pro" # Trying a different model
    else:
        st.session_state.status.error(f":material/error-outline: No suitable Gemini Pro model found in the available models.")
        return None

    if model_name:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.3,
            top_p=0.8
        )
        st.session_state.status.write(f":material/check-circle-outline: {model_name} initialized successfully.")
        return llm
    return None

# Example of how you might use it in your Streamlit app:
if "status" not in st.session_state:
    st.session_state.status = st.empty()

llm = get_llm()

if llm:
    # Proceed with using the llm
    pass
else:
    st.stop() # Stop the app if the LLM couldn't be initialized