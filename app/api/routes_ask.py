from fastapi import APIRouter
from openai import BaseModel

from ..services.rag_pipeline import RAGPipeline
from ..services.vectorstore import VectorStore

router = APIRouter(prefix="/ask", tags=["ask"])

# global store
store = VectorStore(dim=1536)

pipeline = RAGPipeline(store=store)


class AskRequest(BaseModel):
    question: str


@router.post("")
async def ask(req: AskRequest):
    return pipeline.answer(req.question)
