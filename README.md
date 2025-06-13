# Center Desk RAG Assistant

A Retrieval-Augmented Generation (RAG) system for answering Center Desk procedure questions using Streamlit, OpenAI embeddings, and Google's Gemma model.

## Overview

This application retrieves relevant information from a knowledge base and uses a language model to generate accurate answers to questions about Center Desk procedures.

## Features
- Vector-based document retrieval with FAISS
- Context-aware responses using RAG architecture
- Streamlit web interface with streaming responses
- Example prompts to guide users

## Components
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: FAISS
- **LLM**: Google Gemma 2 9B Instruct
- **Frontend**: Streamlit

## Files
- `app.py`: Main application
- `create_vectorstore.py`: Creates vector database from CSV
- `reqs.txt`: Minimal dependencies
- `requirements.txt`: Full dependencies
- `.env`: API keys (not in repo)
- `vector_store/`: FAISS index files

## Setup

1. **Clone and setup environment**
   ```bash
   git clone <repo-url>
   cd center-desk-rag-model
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   pip install -r reqs.txt
   ```

2. **Configure API key**
   Create `.env` file:
   ```
   API_KEY=your_huggingface_api_key
   ```

3. **Create vector store**
   ```bash
   python create_vectorstore.py
   ```

4. **Run application**
   ```bash
   streamlit run app.py
   ```

## Usage
Enter questions about Center Desk procedures in the chat interface. Example queries:
- "How do I forward the desk phone?"
- "How to log packages?"
- "How to close center desk?"

## License
MIT