from fastapi import APIRouter, UploadFile
from app.services.pdf_loader import pdf_reader
import asyncio
from app.services.text_splitter import chunk_text

router = APIRouter(prefix="/upload")


@router.post("")
async def upload_file(file: UploadFile):
    pdf = await asyncio.to_thread(pdf_reader, file.file)
    text = "".join(page.extract_text() for page in pdf.pages)
    chunks = chunk_text(text)
    return {"chunks": chunks[:3], "total_chunks": len(chunks)}
