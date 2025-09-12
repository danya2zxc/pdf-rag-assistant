from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(title=settings.app_name)


@app.get("/ping")
async def ping():
    return {"status": "ok"}
