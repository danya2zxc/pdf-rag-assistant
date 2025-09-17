from unittest.mock import MagicMock, patch

from app.services.rag_pipeline import RAGPipeline


@patch("app.services.rag_pipeline.OpenAI")
def test_rag_pipeline_confidence_threshold(mock_openai):
    mock_store = MagicMock()
    mock_store.search.return_value = [("irrelevant text", 2.0)]  # distance > 1.5

    pipeline = RAGPipeline(store=mock_store, model="gpt-4o-mini")

    result = pipeline.answer("какой уровень учебника?")
    assert result["answer"] == "I don't know"
    assert result["sources"] == []
