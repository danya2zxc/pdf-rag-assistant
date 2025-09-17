from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from openai import OpenAI

from app.core.config import settings


class EmbeddingBase(ABC):
    """
    Abstract base class for embedding models.

    This class defines the interface for embedding models that convert a list of text strings
    into their corresponding vector representations. Subclasses must implement the `embed` method.

    Methods
    -------
    embed(texts: List[str]) -> List[List[float]]:
        Abstract method to generate embeddings for a list of texts. Must be implemented by subclasses.

    Raises
    ------
    NotImplementedError:
        If the `embed` method is not implemented in a subclass.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        raise NotImplementedError


def _iter_batches(items: List[str], batch_size: int):
    """
    Yields successive batches from a list of items.

    Args:
        items (List[str]): The list of string items to be batched.
        batch_size (int): The maximum number of items per batch.

    Yields:
        List[str]: A batch of items with length up to batch_size.
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class EmbeddingOpenAI(EmbeddingBase):
    """
    EmbeddingOpenAI provides text embedding functionality using OpenAI's embedding models.
    Args:
        batch_size (int): Number of texts to process in each batch when requesting embeddings.
        model: The identifier of the OpenAI embedding model to use.
    Methods:
        embed(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of input texts using the specified OpenAI model.
            Texts are stripped of leading/trailing whitespace and processed in batches.
            Returns a list of embedding vectors corresponding to the input texts.
    """

    def __init__(self, batch_size: int, model):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: List[str]):

        texts = [t.strip() for t in texts]

        vectors = []
        for batch in _iter_batches(texts, self.batch_size):

            resp = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend([item.embedding for item in resp.data])
        return vectors


class EmbeddingLocal(EmbeddingBase):
    """
    EmbeddingLocal provides local embedding generation using a specified SentenceTransformer model.
    Args:
        model_name (str): The name or path of the SentenceTransformer model to use.
        batch_size (int): The batch size for encoding texts.
    Raises:
        ImportError: If 'sentence-transformers' or 'torch' are not installed.
    Attributes:
        model (SentenceTransformer): The loaded SentenceTransformer model.
        batch_size (int): The batch size for encoding texts.
    Methods:
        embed(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of input texts.
            Strips whitespace from each text, encodes them in batches,
            normalizes the embeddings, and returns them as lists of floats.
    """

    def __init__(self, model_name: str, batch_size: int):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError(
                "Local embeddings require 'sentence-transformers' and 'torch' installed. "
                "Install with Poetry group or pip before using local backend."
            ) from e

        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed(self, texts: List[str]):
        texts = [t.strip() for t in texts]

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]


@lru_cache(maxsize=1)
def get_embedder() -> EmbeddingBase:
    """
    Initializes and returns an embedding backend instance based on the current settings.

    Returns:
        EmbeddingBase: An instance of the selected embedding backend (EmbeddingOpenAI or EmbeddingLocal).

    Raises:
        ValueError: If the specified embedding backend in settings is not recognized.
    """
    backend = settings.embedding_backend
    if backend == "openai":
        return EmbeddingOpenAI(
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    if backend == "local":
        return EmbeddingLocal(
            model_name=settings.local_embedding_model,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(f"Unknown embedding backend: {backend}")
