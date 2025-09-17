from unittest.mock import MagicMock, patch


@patch("app.services.vectorstore.get_embedder")
def test_vectorstore_add_and_search(mock_get_embedder):
    from app.services.vectorstore import VectorStore

    # mock embeddings: two corpus vectors and one query vector
    mock_emb = MagicMock()
    v = [0.001] * 1536
    mock_emb.embed.side_effect = [
        [v, v],  # add_texts vectors
        [v],  # search query vector
    ]
    mock_get_embedder.return_value = mock_emb

    store = VectorStore(dim=1536)
    store.add_texts(["alpha", "beta"])
    results = store.search("alpha", k=1)

    assert results
    # best match text should be 'alpha'
    assert results[0][0] == "alpha"
