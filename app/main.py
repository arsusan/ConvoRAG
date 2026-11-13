from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.deps import get_vector_store
from app.api.routes import chat, ingestion
from app.core.config import get_settings
from app.db.session import init_db
from app.services.llm import LLMError


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    get_vector_store().ensure_collection()
    yield


app = FastAPI(title=get_settings().app_name, version="1.0.0", lifespan=lifespan)
app.include_router(ingestion.router)
app.include_router(chat.router)


@app.exception_handler(LLMError)
async def llm_error_handler(_: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}