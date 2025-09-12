# PDF RAG Assistant

Pet-project to demonstrate **FastAPI + RAG (Retrieval-Augmented Generation)** skills.  

## Features (planned)
- Upload PDF and extract text
- Chunking and embeddings
- Vector store with FAISS
- Simple RAG pipeline with OpenAI / local models
- REST API: `/upload`, `/ask`, `/docs`
- CI/CD and tests

## Quickstart
```bash
poetry install
poetry run uvicorn app.main:app --reload
