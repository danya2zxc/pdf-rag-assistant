# 📚 PDF RAG Assistant

[![CI](https://github.com/danya2zxc/pfd-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/danya2zxc/pfd-rag-assistant/actions/workflows/ci.yml)

A robust backend service built with **FastAPI** that allows you to chat with your PDF documents. This project implements a complete Retrieval-Augmented Generation (RAG) pipeline, from PDF parsing and text chunking to vector embeddings and answer synthesis with OpenAI's GPT models.

This project is designed to showcase modern backend development practices, including a clean architecture, dependency management with Poetry, containerization with Docker, and automated testing with CI/CD.

---

## ✨ Features

-   **📄 PDF Upload & Processing**: Upload PDF files via a REST API endpoint.
-   **✂️ Smart Text Chunking**: Automatically splits extracted text into semantic chunks using `RecursiveCharacterTextSplitter`.
-   **🧠 Dual Embedding Backends**:
    -   **OpenAI**: Fast and reliable embeddings via `text-embedding-3-small`.
    -   **Local**: High-performance, open-source models (`intfloat/e5-large-v2`) running on local hardware (CPU or GPU).
-   **⚡️ High-Speed Vector Search**: In-memory vector search powered by **FAISS** for instant retrieval of relevant document chunks.
-   **🤖 RAG-Powered Q&A**: Generates accurate, context-aware answers to user questions using OpenAI's `gpt-4o-mini`.
-   **🐳 Dockerized**: Comes with multi-stage `Dockerfile` for production and a `Dockerfile.dev` for local development with hot-reloading.
-   **✅ Tested & Linted**: A comprehensive test suite with `pytest` and a CI pipeline that enforces code quality with `ruff`, `black`, and `isort`.

---

## 🛠️ Tech Stack

| Category         | Technology & Purpose                                                                 |
|------------------|-------------------------------------------------------------------------------------|
| Framework        | **FastAPI** — async backend, all API endpoints and routing                          |
| Config           | **Pydantic Settings** — environment/config management, secrets, runtime options     |
| Text Chunking    | **LangChain** — smart splitting of extracted text into semantic chunks              |
| Vector Search    | **FAISS** — fast similarity search for embeddings                                   |
| Embeddings/LLM   | **OpenAI API**, **Sentence-Transformers** — text embeddings, GPT-based answers      |
| Vector Math      | **numpy** — converts embeddings to float32 arrays for FAISS                         |
| Async Server     | **Uvicorn** — ASGI server for FastAPI                                               |
| Dependencies     | **Poetry** — dependency and environment management                                  |
| Containerization | **Docker** — production and development images                                      |
| Testing          | **Pytest**, **pytest-cov** — test suite and coverage reports                        |
| Code Quality     | **Ruff**, **Black**, **isort** — linting and formatting  
---

## 🏛️ Architecture

The project follows a standard service-based architecture, keeping business logic decoupled from the API layer.

```
.
├── app/
│   ├── api/          # FastAPI routers and request/response models
│   ├── core/         # Application configuration (Pydantic settings)
│   ├── services/     # Business logic (PDF parsing, embeddings, RAG)
│   └── main.py       # FastAPI app entrypoint
├── tests/            # Pytest test suite
├── .github/          # CI workflows and instructions
├── Dockerfile        # Production Docker image
├── Dockerfile.dev    # Development Docker image
└── pyproject.toml    # Project dependencies
```

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.11+
-   [Poetry](https://python-poetry.org/docs/#installation) for dependency management
-   [Docker](https://docs.docker.com/get-docker/) (optional, for containerized setup)

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/danya2zxc/pfd-rag-assistant.git
cd pfd-rag-assistant
poetry install --with dev --without gpu
```

### 2. Configuration

Create a `.env` file in the project root by copying the example:

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY="sk-..."
EMBEDDING_BACKEND="openai" # or "local"
```

**Embedding Backend Options:**
-   `openai` (default): Uses the OpenAI API. Requires `OPENAI_API_KEY`.
-   `local`: Uses a local Sentence Transformers model. For GPU acceleration, install the `gpu` extras:
    ```bash
    poetry install --with dev,gpu
    ```

---

## 🏃‍♀️ Usage

### Running the Server Locally

```bash
poetry run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

### Running with Docker

**For Production:**
```bash
# Build the image
docker build -t pdf-rag-assistant .

# Run the container
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env pdf-rag-assistant
```

**For Development (with hot-reload):**
```bash
# Build the dev image
docker build -f Dockerfile.dev -t pdf-rag-assistant-dev .

# Run the container with a volume mount for live code changes
docker run -p 8000:8000 -v $(pwd)/app:/app/app -v $(pwd)/.env:/app/.env pdf-rag-assistant-dev
```

---

## 🧪 API Endpoints

You can access the interactive API documentation at `http://localhost:8000/docs`.

### 1. Upload a PDF

Upload a PDF file to be processed and indexed. The service will extract text, chunk it, and store the embeddings.

**Request:**
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

**Response:**
```json
{
  "chunks_added": 123
}
```

### 2. Ask a Question

Ask a question about the content of the uploaded PDFs.

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "The main topic of the document is Retrieval-Augmented Generation...",
  "sources": [
    {
      "text": "Retrieval-Augmented Generation (RAG) is a technique...",
      "distance": 0.123
    }
  ]
}
```

---

## ✅ Testing & Quality

This project uses `pytest` for testing and `ruff`/`black` for code quality.

### Running Tests

To run the entire test suite with coverage:

```bash
poetry run pytest --cov=app
```

### Running Linters

To check for formatting and style issues:

```bash
make lint
```

To automatically fix issues:

```bash
make format
```

---

## 🔄 CI/CD

A GitHub Actions workflow is configured in `.github/workflows/ci.yml` to automatically:
1.  **Lint**: Run `make lint` to check code formatting.
2.  **Test**: Execute the `pytest` suite on every push and pull request to the `main` branch.
3.  **Report Coverage**: Uploads the test coverage report as a build artifact.


## 🗺️ Roadmap

- **Full GPU Support:** Implement robust local GPU processing for embeddings and vector search, making the project suitable for large-scale and private deployments.
- **CRAG (Contextual RAG):** Add advanced fallback logic ("I don't know" or custom responses) for cases with low semantic relevance, improving reliability.
- **Vectorstore Enhancements:** Extend vector search to support multi-PDF indexing, metadata (file, page), and more flexible retrieval strategies.
