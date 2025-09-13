from fastapi import APIRouter
from app.services.embeddings import get_embedder

router = APIRouter(prefix="/system", tags=["system"])


@router.post("/embed")
async def embed_texts(texts: list[str]):
    emb = get_embedder()
    vecs = emb.embed(texts)
    return {
        "count": len(vecs),
        "dim": len(vecs[0]) if vecs and vecs[0] else 0,
        "first_vector_preview": vecs[0][:5] if vecs and vecs[0] else []
    }
