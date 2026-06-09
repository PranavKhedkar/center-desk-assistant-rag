import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")


EMBEDDING_MODEL = "text-embedding-3-small"

df = pd.read_csv("center_desk_full_fine_tuning_dataset.csv")

questions = df["input_text"].tolist()
answers = df["target_text"].tolist()
# Each stored document is the full Q+A pair so the LLM receives both the
# canonical question and its answer as context
store_texts = [
    f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)
]

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
# Embed only the questions (not the full Q+A text) as the retrieval key so that
# similarity search matches on question semantics, not answer wording
question_vectors = embeddings.embed_documents(questions)
# FAISS.from_embeddings expects (text_to_store, vector) pairs; the vector is
# the question embedding while the stored text is the full Q+A string
text_embeddings = list(zip(store_texts, question_vectors))

vector_store = FAISS.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=embeddings,
)
vector_store.save_local("./vector_store")
