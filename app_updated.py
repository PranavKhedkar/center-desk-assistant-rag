import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# override=True ensures .env values replace any shell-level env vars already set
load_dotenv(override=True)

VECTOR_STORE_PATH = "./vector_store"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct:scaleway").strip()
# Docs whose relevance score falls below this threshold are dropped before prompting
# the LLM, preventing the model from answering on unrelated context
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
SYSTEM_PROMPT = """You are a Center Desk assistant.

Answer ONLY using the context below.
If the context does not contain the answer, say exactly: "I don't have that procedure in my knowledge base."
Do not invent steps. If you do not know the answer from the context, tell the user to try reaching someone on the duty chain."""


# @st.cache_resource: the client is created once and shared across all Streamlit
# reruns/sessions — avoids reconnecting on every user interaction
@st.cache_resource
def load_inference_client():
    if not HF_TOKEN:
        return None
    return InferenceClient(api_key=HF_TOKEN)


@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


@st.cache_resource
# Leading underscore on _embeddings tells Streamlit's cache not to try hashing
# the embeddings object (it's not hashable); the store is still cached correctly
def load_vector_store(_embeddings):
    try:
        return FAISS.load_local(
            folder_path=VECTOR_STORE_PATH,
            embeddings=_embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None


def build_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}",
        },
    ]


def stream_chat_completion(client: InferenceClient, messages: list[dict]):
    # stream=True yields tokens progressively so st.write_stream can display them
    # as they arrive rather than waiting for the full response
    stream = client.chat.completions.create(
        model=HF_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


inference_client = load_inference_client()
embeddings = load_embeddings()
vector_store = load_vector_store(embeddings)

st.markdown(
    "## Hello! I am a Center Desk Assistant😊\n"
    "How can I assist you with Center Desk procedures today?"
)
st.divider()

st.text("Some example prompts:")
example_prompts = [
    "How do I forward the desk phone?",
    "How to log packages?",
    "How to close center desk?",
]
st.markdown("\n".join(f"- {p}" for p in example_prompts))
st.divider()

user_query = st.chat_input("Ask Here...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    if inference_client is None:
        st.error("HF_TOKEN (or API_KEY) is not set. Add your Hugging Face token to the environment.")
    elif vector_store is None:
        st.error("Vector store is not available. Please check the setup.")
    else:
        try:
            with st.spinner("Retrieving relevant procedures..."):
                results = vector_store.similarity_search_with_relevance_scores(user_query, k=3)
                # Filter out low-confidence results so the LLM isn't prompted with
                # loosely related context that could cause hallucinated procedures
                docs = [doc for doc, score in results if score >= SCORE_THRESHOLD]
                context = "\n".join([doc.page_content for doc in docs])

            messages = build_messages(context, user_query)

            with st.chat_message("assistant"):
                st.write_stream(stream_chat_completion(inference_client, messages))
        except Exception as e:
            st.error(f"An error occurred: {e}")
