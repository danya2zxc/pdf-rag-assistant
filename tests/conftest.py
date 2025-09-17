import importlib
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_API_KEY", "sk-test")


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Provide a dummy OpenAI client for both chat and embeddings
    def _dummy_completion(**kwargs):
        choice = SimpleNamespace(message=SimpleNamespace(content="mocked"))
        return SimpleNamespace(choices=[choice])

    def _dummy_embeddings(**kwargs):
        # return a single 1536-dim vector for each input to match store dim
        inputs = kwargs.get("input") or []
        if isinstance(inputs, str):
            inputs = [inputs]
        vec = [0.001] * 1536
        data = [SimpleNamespace(embedding=vec) for _ in inputs]
        return SimpleNamespace(data=data)

    class DummyOpenAI:
        def __init__(self, *_, **__):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=_dummy_completion)
            )
            self.embeddings = SimpleNamespace(create=_dummy_embeddings)

    # Patch OpenAI in both places where we use it
    monkeypatch.setattr("app.services.rag_pipeline.OpenAI", DummyOpenAI, raising=True)
    monkeypatch.setattr("app.services.embeddings.OpenAI", DummyOpenAI, raising=True)

    app_module = importlib.import_module("app.main")
    return TestClient(app_module.app)
