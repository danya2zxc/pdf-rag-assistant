
from fastapi import APIRouter, UploadFile
from openai import BaseModel

from app.services.vectorstore import VectorStore


router = APIRouter(prefix="/ask", tags=["ask"])

# global store
store = VectorStore(dim=1536)


class AskRequest(BaseModel):
    question: str

@router.post("")
async def ask(req: AskRequest):
    results = store.search(req.question, k=3)
    return {
        "question": req.question,
        "results": [
            {"text": text, "distance": distance}
            for text, distance in results
        ],
    }
