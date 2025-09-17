from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def test_ping(client: TestClient):
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("app.api.routes_system.get_embedder")
def test_system_embed(mock_get_embedder, client: TestClient):
    mock = MagicMock()
    mock.embed.return_value = [[0.001] * 1536]
    mock_get_embedder.return_value = mock

    resp = client.post("/system/embed", json=["hello", "world"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["dim"] == 1536
    assert isinstance(data["first_vector_preview"], list)


@patch("app.services.vectorstore.get_embedder")
def test_ask_endpoint(mock_get_embedder, client: TestClient):
    # mock embedder for both store.add/search and query embedding
    mock_emb = MagicMock()
    mock_emb.embed.side_effect = [
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],  # corpus vectors
        [[1.0, 0.0, 0.0]],  # query vector
    ]
    mock_get_embedder.return_value = mock_emb

    # prepare vector store by calling its add_texts via pipeline store
    from app.api.routes_ask import pipeline

    pipeline.store.add_texts(["alpha text", "beta text"])

    # mock OpenAI chat completion by overwriting pipeline.client
    mock_client = MagicMock()
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="alpha answer"))]
    mock_client.chat.completions.create.return_value = completion

    from app.api.routes_ask import pipeline as _pipeline

    _pipeline.client = mock_client

    resp = client.post("/ask", json={"question": "alpha?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "alpha answer"
    assert isinstance(data.get("sources", []), list)


@patch("app.api.routes_upload.chunk_text")
@patch("app.api.routes_upload.pdf_reader")
def test_upload_endpoint(mock_pdf_reader, mock_chunk_text, client: TestClient):
    # mock pdf reader to return an object with pages and extract_text
    page = MagicMock()
    page.extract_text.return_value = "Hello world from PDF"
    mock_pdf_reader.return_value = MagicMock(pages=[page])

    # mock text splitter result
    mock_chunk_text.return_value = ["Hello world", "from PDF"]

    files = {"file": ("sample.pdf", b"%PDF-1.4\n...", "application/pdf")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_added"] == 2
