import streamlit as st
import time
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# Load environment variables
load_dotenv(override=True)


# Constants
VECTOR_STORE_PATH = "./vector_store"

API_KEY = os.getenv("API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-9b-it",
    task="text-generation",
    huggingfacehub_api_token=API_KEY
)
model = ChatHuggingFace(llm=llm)

# Parser: plain string
parser = StrOutputParser()

client = OpenAI()
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

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


st.markdown(f"## Hello! I am a Center Desk Assistant😊\nHow can I assist you with Center Desk procedures today?")
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

    if len(user_query.split()) < 3:
        with st.chat_message("assistant"):
            st.write("Hello! Please provide a more detailed question about Center Desk procedures.")
    else:
        with st.spinner("Thinking..."):
            # Retrieve relevant docs
            docs = vector_store.similarity_search(user_query, k=3)
            retrieved_context = "\n".join([doc.page_content for doc in docs])

            # Construct prompt for model
            template = PromptTemplate(
            input_variables=["context", "question"],
            template="**Context:**\n{context}\n\n**Question:**\n{question}\n\n**Answer:**\nBased on the context provided, here is the procedure:"
            )

            chain = template | model | parser

            try:
                result = chain.invoke({
                    "context": retrieved_context,
                    "question": user_query
                })
                answer = result.strip()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                answer = "Oops, something went wrong while generating a response."



            with st.chat_message("assistant"):
                st.write_stream(stream(answer))