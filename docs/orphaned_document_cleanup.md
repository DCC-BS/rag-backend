# Orphaned Document Cleanup

This document describes the orphaned document cleanup functionality that helps maintain data consistency between S3 storage and the PostgreSQL database.

## Problem Statement

When files are deleted from S3 buckets (either manually or by external processes), the corresponding document records remain in the PostgreSQL database. This creates "orphaned" documents - database records that reference files that no longer exist in S3 storage.

## Solution

The enhanced document ingestion service now includes functionality to:

1. Check all documents in the database against their corresponding S3 objects
2. Identify orphaned documents (database records without matching S3 files)
3. Remove orphaned documents from the database to maintain consistency

## Usage

### 1. Standalone Cleanup Script

Use the dedicated CLI script for manual cleanup operations:

```bash
# Run cleanup with dry-run to preview what would be deleted
uv run src/rag/cli/cleanup_orphaned_documents.py --dry-run

# Run actual cleanup
uv run src/rag/cli/cleanup_orphaned_documents.py

# Run with verbose logging
uv run src/rag/cli/cleanup_orphaned_documents.py --verbose

# Get help
uv run src/rag/cli/cleanup_orphaned_documents.py --help
```

### 2. Enhanced Ingestion Service

The main ingestion service can now run cleanup during the initial scan:

```bash
# Run ingestion with orphaned document cleanup
uv run src/rag/cli/run_ingestion.py --cleanup-orphaned

# Run initial scan only with cleanup
uv run src/rag/cli/run_ingestion.py --scan-only --cleanup-orphaned

# Use console logging for better readability
uv run src/rag/cli/run_ingestion.py --cleanup-orphaned --console-logging
```

### 3. Programmatic Usage

You can also use the cleanup functionality programmatically:

```python
from rag.services.document_ingestion import S3DocumentIngestionService
from rag.utils.s3 import S3Utils

# Initialize service
service = S3DocumentIngestionService()

# Run standalone cleanup
cleanup_stats = service.cleanup_orphaned_documents()
print(f"Cleaned up {cleanup_stats['deleted_documents']} orphaned documents")

# Run initial scan with cleanup
service.initial_scan(cleanup_orphaned=True)

# You can also use the S3 utility functions directly
s3_utils = S3Utils()

# Check if a specific S3 object exists
exists = s3_utils.object_exists("my-bucket", "path/to/file.pdf")

# Parse an S3 document path
bucket, key = s3_utils.parse_document_path("s3://my-bucket/path/to/file.pdf")
```

## How It Works

### 1. Document Validation Process

1. **Query Database**: Retrieve all document records from PostgreSQL
2. **Parse Paths**: Extract bucket name and object key from each document's S3 path
3. **Check S3 Existence**: Use S3's `head_object` API to verify file existence
4. **Identify Orphans**: Mark documents whose S3 objects return 404 (not found)
5. **Clean Database**: Delete orphaned document records (and their chunks via cascade)

### 2. Statistics Tracking

The cleanup process tracks and reports:

- `total_documents`: Total documents checked
- `orphaned_documents`: Number of orphaned documents found
- `deleted_documents`: Number of documents successfully deleted
- `invalid_paths`: Documents with malformed S3 paths
- `s3_check_errors`: Errors encountered during S3 validation

### 3. Error Handling

- **Invalid Paths**: Documents with malformed S3 paths are logged but not deleted
- **S3 Errors**: Non-404 errors (permissions, connectivity) are logged and skipped
- **Database Errors**: Failed deletions are logged but don't stop the process
- **Atomic Operations**: All deletions are committed together at the end

## Safety Features

### Dry Run Mode

The standalone cleanup script supports dry-run mode to preview changes:

```bash
uv run src/rag/cli/cleanup_orphaned_documents.py --dry-run
```

This will:
- Check all documents against S3
- Report what would be deleted
- Not make any actual changes to the database

### Detailed Logging

All operations are logged with structured logging including:
- Document IDs and file names for orphaned documents
- Detailed error messages for any issues
- Statistics summary at completion

### Conservative Approach

The cleanup process is designed to be conservative:
- Only deletes documents with confirmed 404 responses from S3
- Skips documents with any S3 access errors (better safe than sorry)
- Logs all decisions for audit trails

## Integration with Existing Workflows

### Scheduled Cleanup

You can add cleanup to your regular maintenance schedules:

```bash
# Weekly cleanup (cron example)
0 2 * * 0 cd /path/to/project && uv run src/rag/cli/cleanup_orphaned_documents.py
```

### Before Ingestion

Include cleanup in your ingestion workflows:

```bash
# Clean up orphaned documents before processing new files
uv run src/rag/cli/run_ingestion.py --scan-only --cleanup-orphaned
```

### Monitoring Integration

The cleanup statistics can be integrated into monitoring systems:

```python
import json
from rag.services.document_ingestion import S3DocumentIngestionService

service = S3DocumentIngestionService()
stats = service.cleanup_orphaned_documents()

# Send to monitoring system
print(json.dumps(stats, indent=2))
```

## Performance Considerations

- **S3 API Calls**: Each document requires one S3 HEAD request
- **Batch Processing**: Database operations are batched for efficiency
- **Memory Usage**: All documents are loaded into memory for processing
- **Network Latency**: Performance depends on S3 response times

For large databases, consider running cleanup during off-peak hours.

## Troubleshooting

### Common Issues

1. **S3 Connectivity**: Ensure S3 credentials and endpoint are properly configured
2. **Database Permissions**: Verify the service has DELETE permissions on document tables
3. **Large Datasets**: For very large document sets, consider adding progress indicators

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
uv run src/rag/cli/cleanup_orphaned_documents.py --verbose
```

This provides detailed information about each document validation step.

## Utility Functions

The cleanup functionality is built on reusable S3 utility functions that are available in the `S3Utils` class:

### `object_exists(bucket_name, object_key)`

Checks if an object exists by performing a HEAD request.

```python
from rag.utils.s3 import S3Utils

s3_utils = S3Utils()
exists = s3_utils.object_exists("documents-el", "reports/quarterly.pdf")
if exists:
    print("File exists in S3")
else:
    print("File not found in S3")
```

**Returns:**
- `True` if the object exists
- `False` if the object is not found (404)

**Raises:**
- `ClientError` for S3 errors other than 404 (permissions, connectivity, etc.)

### `parse_document_path(document_path)`

Parses a document path to extract bucket name and object key.

```python
from rag.utils.s3 import S3Utils

s3_utils = S3Utils()
result = s3_utils.parse_document_path("s3://documents-el/reports/quarterly.pdf")
if result:
    bucket_name, object_key = result
    print(f"Bucket: {bucket_name}, Key: {object_key}")
else:
    print("Invalid S3 path format")
```

**Input format:** `s3://bucket-name/object-key`

**Returns:**
- `tuple[str, str]` - (bucket_name, object_key) if valid
- `None` if the path format is invalid

These utilities can be used independently for other S3-related operations throughout the codebase.

## Future Enhancements

Potential improvements could include:
- Progress bars for large cleanup operations
- Selective cleanup by bucket or access role
- Backup creation before deletion
- Integration with S3 event notifications for real-time cleanup
