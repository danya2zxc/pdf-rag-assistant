from fastapi import FastAPI

from .api.routes_ask import router as ask_router
from .api.routes_system import router as system_router
from .api.routes_upload import router as upload
from .core.config import settings

app = FastAPI(title=settings.app_name)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


app.include_router(upload)
app.include_router(system_router)
app.include_router(ask_router)
