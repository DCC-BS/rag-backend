import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer
from fastapi_azure_auth.user import User

from rag.models.api_models import (
    ChatMessage,
    DocumentListResponse,
    DocumentMetadata,
    DocumentOperationResponse,
)
from rag.services.document_management import DocumentManagementService
from rag.services.rag_pipeline import SHRAGPipeline
from rag.utils.config import ConfigurationManager
from rag.utils.logger import get_logger, init_logger
from rag.utils.usage_tracking import get_pseudonymized_user_id

init_logger()
logger = get_logger()


CONST_SPLIT_STRING = "\0"

config = ConfigurationManager.get_config()
scope_name = f"api://{config.AZURE_CLIENT_ID}/{config.SCOPE_DESCRIPTION}"
scopes = {
    scope_name: config.SCOPE_DESCRIPTION,
}
azure_scheme = SingleTenantAzureAuthorizationCodeBearer(
    app_client_id=config.AZURE_CLIENT_ID,
    tenant_id=config.AZURE_TENANT_ID,
    scopes=scopes,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load OpenID config on startup.
    """
    await azure_scheme.openid_config.load_config()
    app.state.pipeline = SHRAGPipeline()
    app.state.document_service = DocumentManagementService()
    yield


def get_pipeline(request: Request) -> SHRAGPipeline:
    return request.app.state.pipeline


def get_document_service(request: Request) -> DocumentManagementService:
    return request.app.state.document_service


def get_app() -> FastAPI:
    app: FastAPI = FastAPI(
        title="RAG API",
        lifespan=lifespan,
    )

    origins: list[str] = config.CORS_ORIGINS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = get_app()


async def validate_is_writer_user(user: Annotated[User, Depends(azure_scheme)]) -> User:
    """
    Validate that a user is in the `Writer` role in order to access the API.
    Raises a 401 authentication error if not.
    """
    if "Writer" not in user.roles:
        raise HTTPException(status_code=401, detail="User is has no `Writer` role")
    return user


Writer = Annotated[User, Depends(validate_is_writer_user)]


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(azure_scheme)]):
    return current_user


@app.post("/chat")
async def chat(
    chat_message: ChatMessage,
    current_user: Annotated[User, Depends(azure_scheme)],
    pipeline: Annotated[SHRAGPipeline, Depends(get_pipeline)],
) -> StreamingResponse:
    pseudonym_id = get_pseudonymized_user_id(current_user, config.HMAC_SECRET)
    logger.info("app_event", extra={"pseudonym_id": pseudonym_id, "event": "chat_message"})

    async def event_generator() -> AsyncGenerator[str, Any]:
        try:
            async for event in pipeline.astream_query(
                message=chat_message.message,
                user_organizations=current_user.roles,
                thread_id=chat_message.thread_id,
                document_ids=chat_message.document_ids,
            ):
                yield f"{event.model_dump_json()}{CONST_SPLIT_STRING}"

        except Exception as e:
            logger.error(
                "Error processing stream query",
                extra={
                    "error": str(e),
                },
                exc_info=True,
            )
            yield "Ein Fehler ist aufgetreten. Bitte versuchen Sie es später erneut."

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# Document Management Endpoints


@app.get("/documents", response_model=DocumentListResponse)
async def search_documents(
    current_user: Annotated[User, Depends(azure_scheme)],
    document_service: Annotated[DocumentManagementService, Depends(get_document_service)],
    query: str | None = None,
    limit: int = 5,
) -> DocumentListResponse:
    """Search documents by query or get all documents accessible by the current user."""
    if query is None:
        documents_data = document_service.get_user_documents(current_user.roles)
    else:
        pseudonym_id = get_pseudonymized_user_id(current_user, config.HMAC_SECRET)
        logger.info("app_event", extra={"pseudonym_id": pseudonym_id, "event": "search_documents"})
        documents_data = document_service.search_documents(query, current_user.roles, limit)

    documents = [DocumentMetadata(**doc) for doc in documents_data]
    return DocumentListResponse(documents=documents, total_count=len(documents))


@app.get("/documents/{document_id}")
async def get_document_by_id(
    document_id: int,
    current_user: Annotated[User, Depends(azure_scheme)],
    document_service: Annotated[DocumentManagementService, Depends(get_document_service)],
) -> Response:
    """Get document content by ID."""
    content = document_service.get_document_by_id(document_id, current_user.roles)

    # Return the file content as binary response
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=document_{document_id}"},
    )


@app.post("/documents", response_model=DocumentOperationResponse)
async def upload_document(
    access_role: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    current_user: Writer,
    document_service: Annotated[DocumentManagementService, Depends(get_document_service)],
) -> DocumentOperationResponse:
    """Upload a new document to S3."""
    pseudonym_id = get_pseudonymized_user_id(current_user, config.HMAC_SECRET)
    logger.info("app_event", extra={"pseudonym_id": pseudonym_id, "event": "upload_document"})

    results: dict[str, list[str]] = {"failed": [], "success": []}
    # Only raise an exception if we are handling single file upload
    raise_exception: bool = len(files) == 1
    for file in files:
        try:
            _ = document_service.upload_document(file, access_role, current_user.roles)
            results["success"].append(file.filename or "")
        except Exception:
            results["failed"].append(file.filename or "")
            if raise_exception:
                raise

    message = f"{len(results["success"])} files uploaded successfully. {len(results["failed"])} files failed to upload."
    return DocumentOperationResponse(
        message=message,
        file_name="",
        additional_info={
            "success": len(results["success"]),
            "failed": len(results["failed"]),
            "failed_files": results["failed"],
        },
    )


@app.put("/documents/{document_id}", response_model=DocumentOperationResponse)
async def update_document(
    document_id: int,
    access_role: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    current_user: Writer,
    document_service: Annotated[DocumentManagementService, Depends(get_document_service)],
) -> DocumentOperationResponse:
    """Update an existing document in S3."""
    pseudonym_id = get_pseudonymized_user_id(current_user, config.HMAC_SECRET)
    logger.info("app_event", extra={"pseudonym_id": pseudonym_id, "event": "update_document"})

    result = document_service.update_document(document_id, file, access_role, current_user.roles)

    return DocumentOperationResponse(
        message=result["message"],
        document_id=str(result["document_id"]),
        file_name=result["file_name"],
        additional_info={"original_filename": result["original_filename"], "size": result["size"]},
    )


@app.delete("/documents/{document_id}", response_model=DocumentOperationResponse)
async def delete_document(
    document_id: int,
    current_user: Writer,
    document_service: Annotated[DocumentManagementService, Depends(get_document_service)],
) -> DocumentOperationResponse:
    """Delete a document from S3 and database."""
    pseudonym_id = get_pseudonymized_user_id(current_user, config.HMAC_SECRET)
    logger.info("app_event", extra={"pseudonym_id": pseudonym_id, "event": "delete_document"})

    result = document_service.delete_document(document_id, current_user.roles)

    return DocumentOperationResponse(
        message=result["message"], document_id=result["document_id"], file_name=result["file_name"]
    )


if __name__ == "__main__":
    host: str = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port: int = int(os.environ.get("BACKEND_PORT", "8080"))
    reload: bool = os.environ.get("BACKEND_DEV", "False").lower() == "true"
    uvicorn.run("src.rag.app:app", host=host, port=port, reload=reload)
