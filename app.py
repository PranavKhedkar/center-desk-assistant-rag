import streamlit as st
import time
import requests
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Constants
VECTOR_STORE_PATH = "./vector_store"
MODEL_API_ENDPOINT = os.getenv("MODEL_API_ENDPOINT")  
API_KEY = os.getenv("API_KEY")

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = FAISS.load_local(
    folder_path=VECTOR_STORE_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

# Streaming output
def stream(text, delay: float = 0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# UI
st.markdown("## Hello! 😊\nHow can I assist you with Center Desk procedures today?")
st.divider()

st.text("Some example prompts:\n")
example_prompts = [
    'How do I forward the desk phone?',
    'How to log packages?',
    'How to close center desk?'
]
st.markdown('\n'.join(f"- {p}" for p in example_prompts))
st.divider()

# Chat input
user_query = st.chat_input("Ask Here...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    if len(user_query.split()) < 4:
        with st.chat_message("assistant"):
            st.write("Hello! How can I assist you with Center Desk procedures today?")
    else:
        with st.spinner("Thinking..."):
            # Retrieve relevant docs
            docs = vector_store.similarity_search(user_query, k=3)
            retrieved_context = "\n".join([doc.page_content for doc in docs])

            # Construct prompt for model
            prompt = (
                f"Context: {retrieved_context}\n\n"
                f"Question: {user_query}\n\n"
                f"Answer:"
            )

            # Hugging Face API request
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            payload = {
                "inputs": prompt,
                "options": {"use_cache": False}
            }

            try:
                response = requests.post(MODEL_API_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()

                # Extract answer (some models return a list, some a dict)
                result = response.json()
                if isinstance(result, list):
                    answer = result[0].get("generated_text", "No answer found.")
                elif isinstance(result, dict) and "generated_text" in result:
                    answer = result["generated_text"]
                else:
                    answer = str(result)  # fallback for unknown structure

            except Exception as e:
                answer = f"Error: Could not get response from model API.\nDetails: {e}"

        with st.chat_message("assistant"):
            st.write_stream(stream(answer))
