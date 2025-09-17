import asyncio

from fastapi import APIRouter, UploadFile

from app.api.routes_ask import store
from app.services.pdf_loader import pdf_reader
from app.services.text_splitter import chunk_text

router = APIRouter(prefix="/upload")


@router.post("")
async def upload_file(file: UploadFile):
    """Endpoint to upload a PDF and return text chunks"""
    # Read the PDF file
    pdf = await asyncio.to_thread(pdf_reader, file.file)
    # Extract text and chunk it
    text = "".join(page.extract_text() for page in pdf.pages)
    chunks = chunk_text(text)

    store.add_texts(chunks)
    # Return the first 3 chunks and total count for verification
    return {"chunks_added": len(chunks)}
