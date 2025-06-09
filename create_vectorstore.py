import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# 1. Load the CSV file
df = pd.read_csv('center_desk_full_fine_tuning_dataset.csv')

# 2. Create LangChain Documents from the CSV data
docs = []
for _, row in df.iterrows():
    content = f"Question: {row['input_text']}\nAnswer: {row['target_text']}"
    doc = Document(page_content=content, metadata={})
    docs.append(doc)

# 3. Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# 4. Create and save the FAISS vector store
vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local("./vector_store")
