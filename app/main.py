from fastapi import FastAPI
from app.core.config import settings
from .api.routes_upload import router as upload 
from app.api.routes_system import router as system_router
app = FastAPI(title=settings.app_name)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


app.include_router(upload)
app.include_router(system_router)
