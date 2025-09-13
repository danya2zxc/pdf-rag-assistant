
from typing import List
from app.core.config import settings
from openai import OpenAI
from abc import ABC, abstractmethod
from functools import lru_cache




class EmbeddingBase(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        raise NotImplementedError

def _iter_batches(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class EmbeddingOpenAI(EmbeddingBase):
    def __init__(self, batch_size: int, model):
        self.client = OpenAI()
        self.model =  model
        self.batch_size = batch_size
    
    def embed(self, texts: List[str]):
        
        texts = [t.strip() for t in texts]
        
        vectors = []
        for batch in _iter_batches(texts, self.batch_size):
            
            resp = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            vectors.extend([item.embedding for item in resp.data])
        return vectors
    
class EmbeddingLocal(EmbeddingBase):
    def __init__(self, model_name: str, batch_size: int):
        try:
            from sentence_transformers import SentenceTransformer
            import torch 
        except Exception as e:
            raise ImportError(
                "Local embeddings require 'sentence-transformers' and 'torch' installed. "
                "Install with Poetry group or pip before using local backend."
            ) from e

        from sentence_transformers import SentenceTransformer
        import torch

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
