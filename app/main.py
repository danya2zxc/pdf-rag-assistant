from fastapi import FastAPI
from app.core.config import settings
from .api import routes_upload as upload
app = FastAPI(title=settings.app_name)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


app.include_router(upload.router)
