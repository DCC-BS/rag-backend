# Document Management API

This document describes the document management API endpoints that allow users to perform CRUD operations on documents stored in S3 buckets based on their access roles.

## Authentication

All endpoints require authentication via Bearer token. The authenticated user's access roles determine which documents they can access and which operations they can perform.

## Endpoints

### 1. Get User Documents

**GET /documents**

Returns a list of all documents accessible by the current user based on their access roles.

**Response:**
```json
{
  "documents": [
    {
      "id": 1,
      "file_name": "example.pdf",
      "document_path": "s3://bucket-role1/example.pdf",
      "mime_type": "application/pdf",
      "num_pages": 10,
      "created_at": "2024-01-15T10:30:00Z",
      "access_roles": ["ROLE1"]
    }
  ],
  "total_count": 1
}
```

### 2. Get Document by ID

**GET /documents/{document_id}**

Downloads the actual document file from S3 by document ID.

**Parameters:**
- `document_id` (path): ID of the document to download

**Response:**
- Binary file content with appropriate headers for file download
- Returns 404 if document not found
- Returns 403 if user doesn't have access to the document

### 3. Upload Document

**POST /documents**

Uploads a new document to S3 bucket associated with the specified access role.

**Request Body (multipart/form-data):**
- `access_role` (form field): Target access role for the document
- `file` (file): Document file to upload

**Response:**
```json
{
  "message": "File uploaded successfully",
  "file_name": "example.pdf",
  "additional_info": {
    "bucket_name": "bucket-role1",
    "object_key": "example.pdf",
    "normalized_filename": "example.pdf",
    "size": 12345
  }
}
```

### 4. Update Document

**PUT /documents/{document_id}**

Updates an existing document by uploading a new version to S3.

**Parameters:**
- `document_id` (path): ID of the document to update

**Request Body (multipart/form-data):**
- `access_role` (form field): Access role context for the operation
- `file` (file): New document file content

**Response:**
```json
{
  "message": "Document updated successfully",
  "document_id": "1",
  "file_name": "example.pdf",
  "additional_info": {
    "original_filename": "example_updated.pdf",
    "size": 13456
  }
}
```

### 5. Delete Document

**DELETE /documents/{document_id}**

Deletes a document from both S3 and the database.

**Parameters:**
- `document_id` (path): ID of the document to delete

**Request Body:**
```json
{
  "access_role": "ROLE1"
}
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "1",
  "file_name": "example.pdf"
}
```

## Access Control

- Users can only access documents that have at least one access role in common with their assigned roles
- For upload operations, users must have the specific access role they're uploading to
- For update/delete operations, users must have the specific access role and the document must be assigned to that role
- All operations validate user permissions before proceeding

## File Processing

- Uploaded files are automatically normalized (special characters, umlauts, spaces converted to underscores)
- Files are stored in S3 buckets
- The document ingestion service will automatically process uploaded files for search indexing
- Updated documents will be reprocessed by the ingestion service when detected

## Error Responses

- `400 Bad Request`: Invalid request parameters or missing file
- `401 Unauthorized`: Authentication required or invalid token
- `403 Forbidden`: Access denied to specified role or document
- `404 Not Found`: Document not found
- `500 Internal Server Error`: Server-side processing error
