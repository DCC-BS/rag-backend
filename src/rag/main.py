import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Annotated, Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session

from rag.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    Token,
    authenticate_user,
    create_access_token,
    get_current_user,
)
from rag.core.rag_pipeline import SHRAGPipeline
from rag.models import User, create_db_and_tables, get_session

logger: structlog.stdlib.BoundLogger = structlog.get_logger()

state: dict[str, SHRAGPipeline] = {}

TOKEN_TYPE = os.environ.get("TOKEN_TYPE") or "bearer"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # pyright: ignore[reportUnusedParameter]
    create_db_and_tables()
    state["pipeline"] = SHRAGPipeline()
    yield


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


@app.get("/stream_query")
async def stream_query(
    question: str,
    current_user: Annotated[User, Depends(get_current_user)],
    thread_id: str = "default",
) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, Any]:
        try:
            async for event in state["pipeline"].astream_query(
                question,
                current_user.organization,
                thread_id,
            ):
                yield f"{event}\n"

        except StopAsyncIteration as e:
            # Handle the intentional generator stop with return value
            if hasattr(e, "value"):
                yield f"{e.value}\n"  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        except Exception as e:
            logger.error(
                "Error processing stream query",
                extra={
                    "user": current_user.username,
                    "thread_id": thread_id,
                    "question": question,
                    "error": str(e),
                },
                exc_info=True,
            )
            yield "Ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut.\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
    )


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: Annotated[Session, Depends(get_session)],
) -> Token:
    user = authenticate_user(
        username=form_data.username,
        password=form_data.password,
        session=session,
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token: str = create_access_token(
        data={"username": user.username},
        expires_delta=access_token_expires,
    )
    return Token(
        access_token=access_token,
        token_type=TOKEN_TYPE,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
