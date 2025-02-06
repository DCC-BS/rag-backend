from contextlib import asynccontextmanager
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Generator,
    Literal,
)

import bcrypt
import jwt
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.applications import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from langchain.schema import Document
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select
from starlette.responses import StreamingResponse

from core.rag_pipeline import SHRAGPipeline


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True)
    hashed_password: bytes
    organization: str
    disabled: bool = False


state: dict[str, SHRAGPipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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


SECRET_KEY = "81e7ad176e1d9e40d3c467791f4b7b11b76700e9ad9f5e3d119f3537d11e4b46"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(
    plain_password,
    hashed_password,
) -> bool:
    password_byte_enc = plain_password.encode("utf-8")
    return bcrypt.checkpw(password=password_byte_enc, hashed_password=hashed_password)


def get_password_hash(
    password: str,
) -> bytes:
    pwd_bytes: bytes = password.encode(encoding="utf-8")
    salt: bytes = bcrypt.gensalt()
    hashed_password: bytes = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def get_user(
    username: str,
    session: SessionDep,
) -> User | None:
    user: User | None = session.exec(
        select(User).where(User.username == username)
    ).first()
    if user:
        return user
    return None


def authenticate_user(
    username: str,
    password: str,
    session: SessionDep,
) -> Any | Literal[False]:
    user: Any = get_user(username=username, session=session)
    if not user:
        return False
    if not verify_password(
        plain_password=password,
        hashed_password=user.hashed_password,
    ):
        return False

    return user


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    to_encode: dict[Any, Any] = data.copy()
    expire_time: datetime
    if expires_delta:
        expire_time = datetime.now(tz=timezone.utc) + expires_delta
    else:
        expire_time = datetime.now(tz=timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire_time})

    encoded_jwt: str = jwt.encode(
        payload=to_encode,
        key=SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return encoded_jwt


async def get_current_user(
    token: Annotated[
        str,
        Depends(dependency=oauth2_scheme),
    ],
    session: SessionDep,
) -> Any:
    credentials_exception: HTTPException = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: Any = jwt.decode(
            jwt=token,
            key=SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        username: str = payload.get("username")
        if username is None:
            raise credentials_exception
        token_data: TokenData = TokenData(username=username)

    except InvalidTokenError as err:
        raise credentials_exception from err
    if token_data.username is None:
        raise credentials_exception
    user: Any = get_user(username=token_data.username, session=session)
    if user is None:
        raise credentials_exception

    return user


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    context: list[
        dict[
            str,
            Any,
        ]
    ]


def document_to_dict(
    doc: Document,
) -> dict[
    str,
    Any,
]:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    current_user: Annotated[
        User,
        Depends(get_current_user),
    ],
):
    """
    POST endpoint to run a query through the RAG pipeline synchronously.
    """
    try:
        (
            answer,
            context,
        ) = state["pipeline"].query(
            request.question,
            current_user.organization,
            "default",
        )
        context_dict: list[dict[str, Any]] = [document_to_dict(doc) for doc in context]
        return {
            "answer": answer,
            "context": context_dict,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


# @app.get("/stream_query")
# async def stream_query(
#     question: str,
#     current_user: Annotated[
#         User,
#         Depends(get_current_user),
#     ],
#     thread_id: str = "default",
# ) -> StreamingResponse:
#     """
#     GET endpoint to stream query events over text. This leverages the
#     asynchronous functionality provided by the pipeline.
#     """

#     async def event_generator() -> AsyncGenerator[str, Any]:
#         try:
#             async for event in pipeline.astream_query(
#                 question,
#                 current_user.organization,
#                 thread_id,
#             ):
#                 yield f"{event}\n"

#         except Exception as e:
#             yield f"Error: {str(e)}\n"

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/plain",
#     )


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[
        OAuth2PasswordRequestForm,
        Depends(),
    ],
    session: SessionDep,
) -> Token:
    user: Any = authenticate_user(
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
        token_type="bearer",
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
    )
