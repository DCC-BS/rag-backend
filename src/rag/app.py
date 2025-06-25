import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag.auth import get_current_user
from rag.models.user import User
from rag.services.rag_pipeline import SHRAGPipeline

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


TOKEN_TYPE = os.environ.get("TOKEN_TYPE") or "bearer"
CONST_SPLIT_STRING = "\0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.pipeline = SHRAGPipeline()
    yield


def get_pipeline(request: Request) -> SHRAGPipeline:
    return request.app.state.pipeline


app: FastAPI = FastAPI(title="RAG API", lifespan=lifespan)

origins: list[str] = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://localhost:50001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    # If code reaches here, the token was valid.
    # current_user contains the validated user info.
    return current_user


class ChatMessage(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/chat")
async def chat(
    chat_message: ChatMessage,
    current_user: Annotated[User, Depends(get_current_user)],
    pipeline: Annotated[SHRAGPipeline, Depends(get_pipeline)],
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, Any]:
        try:
            async for event in pipeline.astream_query(
                chat_message.message,
                current_user.organizations,
                chat_message.thread_id,
            ):
                yield f"{event.model_dump_json()}{CONST_SPLIT_STRING}"

        except Exception as e:
            logger.error(
                "Error processing stream query",
                extra={
                    "user": current_user.username,
                    "thread_id": chat_message.thread_id,
                    "message": chat_message.message,
                    "error": str(e),
                },
                exc_info=True,
            )
            yield "Ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut."

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    host: str = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port: int = int(os.environ.get("BACKEND_PORT", "8080"))
    reload: bool = os.environ.get("BACKEND_DEV", "False").lower() == "true"
    uvicorn.run("src.rag.app:app", host=host, port=port, reload=reload)
