import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv(override=True)

# Constants
VECTOR_STORE_PATH = "./vector_store"
HUGGINGFACE_API_KEY = os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MIN_QUERY_WORDS = 3

# Initialize models
# @st.cache_resource
def load_models():
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-2-9b-it",
        task="text-generation",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        streaming=True,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    return model, embeddings

# @st.cache_resource
def load_vector_store(embeddings):
    try:
        return FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

model, embeddings = load_models()
vector_store = load_vector_store(embeddings)

# UI
st.markdown("## Hello! I am a Center Desk Assistant😊\nHow can I assist you with Center Desk procedures today?")
st.divider()

st.text("Some example prompts:")
example_prompts = [
    'How do I forward the desk phone?',
    'How to log packages?',
    'How to close center desk?'
]
st.markdown('\n'.join(f"- {p}" for p in example_prompts))
st.divider()

# Chat interaction
user_query = st.chat_input("Ask Here...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    if len(user_query.split()) < MIN_QUERY_WORDS:
        with st.chat_message("assistant"):
            st.write("Please provide a more detailed question about Center Desk procedures.")
    elif vector_store is None:
        st.error("Vector store is not available. Please check the setup.")
    else:
        try:
            with st.spinner("Retrieving relevant procedures..."):
                docs = vector_store.similarity_search(user_query, k=3)
                context = "\n".join([doc.page_content for doc in docs])

            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="**Context:**\n{context}\n\n**Question:**\n{question}\n\n**Answer:**\nBased on the context provided, here is the procedure:"
            )
            chain = prompt | model | StrOutputParser()

            with st.chat_message("assistant"):
                st.write_stream(
                    chain.stream({
                        "context": context,
                        "question": user_query,
                    })
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")