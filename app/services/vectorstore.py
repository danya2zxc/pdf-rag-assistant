
from typing import List
import faiss

import numpy as np
from app.services.embeddings import get_embedder

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self.embedder = get_embedder()

    def add_texts(self, texts: List[str]):
        vectors = self.embedder.embed(texts)
        vectors_np = np.array(vectors).astype("float32")
        
        self.index.add(vectors_np)
        self.texts.extend(texts)

    def search(self, query: str, k: int = 3):
        q_vec = self.embedder.embed([query])
        q_np = np.array(q_vec).astype("float32")

        distance, indices = self.index.search(q_np, k)

        results = []
        
        for idx, dist in zip(indices[0], distance[0]):
            if idx != -1:
                results.append((self.texts[idx], float(dist)))
        return results
