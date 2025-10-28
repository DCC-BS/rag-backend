"""S3 utilities for MinIO/S3 operations."""
# pyright: reportMissingTypeStubs=false

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3  # type: ignore[reportMissingTypeStubs]
from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore[reportMissingTypeStubs]
from structlog.stdlib import BoundLogger

from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.logger import get_logger


class S3Utils:
    """Utility class for S3/MinIO operations."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize S3 utilities with configuration."""
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger: BoundLogger = get_logger()
        self.s3_client: Any = self._setup_client()

    def _setup_client(self) -> Any:
        """Set up and configure the S3 client for MinIO."""
        try:
            client = boto3.client(
                "s3",
                endpoint_url=self.config.INGESTION.S3_ENDPOINT,
                aws_access_key_id=self.config.INGESTION.S3_ACCESS_KEY,
                aws_secret_access_key=self.config.INGESTION.S3_SECRET_KEY,
                region_name="us-east-1",  # MinIO default
            )
            # Test connection
            client.list_buckets()
        except NoCredentialsError:
            self.logger.exception("S3 credentials not provided")
            raise
        except ClientError as e:
            self.logger.exception("Failed to connect to S3/MinIO", error=str(e))
            raise
        except Exception as e:
            self.logger.exception("Unexpected error connecting to S3/MinIO", error=str(e))
            raise
        else:
            self.logger.info("Successfully connected to S3/MinIO", endpoint=self.config.INGESTION.S3_ENDPOINT)
            return client

    def get_bucket_name(self) -> str:
        """Get the single bucket name from environment variable.

        Defaults to "rag-bot" when S3_BUCKET_NAME is not set.
        """
        return os.environ.get("S3_BUCKET_NAME", "rag-bot")

    def ensure_bucket_exists(self, bucket_name: str) -> None:
        """Ensure a bucket exists, creating it if necessary."""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Bucket doesn't exist, create it
                try:
                    self.s3_client.create_bucket(Bucket=bucket_name)
                    self.logger.info(f"Created bucket {bucket_name}")
                except ClientError as create_error:
                    self.logger.exception(f"Failed to create bucket {bucket_name}", error=str(create_error))
                    raise
            else:
                self.logger.exception(f"Error checking bucket {bucket_name}", error=str(e))
                raise

    def ensure_main_bucket_exists(self) -> None:
        """Ensure the single application bucket exists."""
        bucket_name = self.get_bucket_name()
        self.ensure_bucket_exists(bucket_name)

    def format_s3_path(self, bucket_name: str, object_key: str) -> str:
        """Format S3 path for consistent logging."""
        return f"s3://{bucket_name}/{object_key}"

    def extract_access_role_from_object_key(self, object_key: str) -> str | None:
        """Extract access role from the first directory segment of the object key."""
        parts = object_key.split("/", 1)
        if len(parts) > 1 and parts[0]:
            return parts[0].strip().upper()
        return None

    def get_tags(self, bucket_name: str, object_key: str) -> dict[str, str]:
        """Get object tags as a dictionary."""
        try:
            response = self.s3_client.get_object_tagging(Bucket=bucket_name, Key=object_key)
            return {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}
        except ClientError:
            return {}

    def update_tags(self, bucket_name: str, object_key: str, tag_set: list[dict[str, str]]) -> None:
        """Update object tags."""
        try:
            self.s3_client.put_object_tagging(Bucket=bucket_name, Key=object_key, Tagging={"TagSet": tag_set})
        except ClientError as e:
            self.logger.warning(f"Failed to update tags for {object_key}", error=str(e))

    def upload_file(self, local_path: Path, bucket_name: str, object_key: str) -> bool:
        """Upload a single file to S3."""
        try:
            self.s3_client.upload_file(str(local_path), bucket_name, object_key)
        except ClientError as e:
            self.logger.exception(f"Failed to upload {local_path}", error=str(e))
            return False
        else:
            self.logger.info(f"Uploaded {local_path} to s3://{bucket_name}/{object_key}")
            return True

    def download_object(self, bucket_name: str, object_key: str) -> bytes:
        """Download an object and return its content."""
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=object_key)
            return response["Body"].read()
        except ClientError as e:
            self.logger.exception(f"Failed to download {object_key} from {bucket_name}", error=str(e))
            raise

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """Delete an object."""
        s3_path = self.format_s3_path(bucket_name, object_key)
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_key)
            self.logger.info(f"Deleted file from S3: {s3_path}")
        except ClientError as e:
            self.logger.exception(f"Failed to delete S3 object {s3_path}", error=str(e))
            raise

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        """Check if an object exists.

        Args:
            bucket_name: The S3 bucket name
            object_key: The S3 object key

        Returns:
            True if the object exists, False if not found

        Raises:
            ClientError: For S3 errors other than 404 (not found)
        """
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=object_key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            # Re-raise other errors (permissions, etc.)
            self.logger.warning(f"Error checking S3 object existence: {bucket_name}/{object_key}", error=str(e))
            raise
        else:
            return True

    def parse_document_path(self, document_path: str) -> tuple[str, str] | None:
        """Parse document path to extract bucket name and object key.

        Args:
            document_path: S3 path in format s3://bucket-name/object-key

        Returns:
            Tuple of (bucket_name, object_key) or None if invalid format
        """
        if not document_path.startswith("s3://"):
            self.logger.warning(f"Invalid document path format: {document_path}")
            return None

        path_parts = document_path[5:].split("/", 1)  # Remove s3:// prefix
        if len(path_parts) != 2:
            self.logger.warning(f"Invalid document path format: {document_path}")
            return None

        return path_parts[0], path_parts[1]

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize path for S3 by replacing umlauts and removing problematic characters."""
        # Replace German umlauts
        umlaut_replacements = {"ä": "ae", "Ä": "Ae", "ü": "ue", "Ü": "Ue", "ö": "oe", "Ö": "Oe", "ß": "ss"}

        normalized = path
        for umlaut, replacement in umlaut_replacements.items():
            normalized = normalized.replace(umlaut, replacement)

        # Remove or replace spaces with underscores
        normalized = normalized.replace(" ", "_")

        # Remove or replace other problematic characters
        # Keep only alphanumeric, underscores, hyphens, dots, and forward slashes
        normalized = re.sub(r"[^a-zA-Z0-9._/-]", "_", normalized)

        # Clean up multiple consecutive underscores
        normalized = re.sub(r"_{2,}", "_", normalized)

        # Remove leading/trailing underscores from filename parts
        parts = normalized.split("/")
        parts = [part.strip("_") for part in parts if part.strip("_")]
        normalized = "/".join(parts)

        return normalized


class S3DocumentTagger:
    """Utility class for managing S3 document tags related to processing status."""

    def __init__(self, s3_utils: S3Utils, config: AppConfig | None = None) -> None:
        """Initialize S3 document tagger."""
        self.s3_utils: S3Utils = s3_utils
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger: BoundLogger = get_logger()

        # Error tags for document tagging
        self.error_tags: set[str] = {
            self.config.INGESTION.UNPROCESSABLE_TAG,
            self.config.INGESTION.NOT_SUPPORTED_FILE_TAG,
            self.config.INGESTION.NO_CHUNKS_TAG,
        }
        self.all_processing_tags: set[str] = self.error_tags | {self.config.INGESTION.PROCESSED_TAG}

    def is_document_processed(self, bucket_name: str, object_key: str) -> bool:
        """Check if a document has been processed by looking for the processed tag."""
        tags = self.s3_utils.get_tags(bucket_name, object_key)
        return self.config.INGESTION.PROCESSED_TAG in tags

    def has_error_tag(self, bucket_name: str, object_key: str) -> bool:
        """Check if a document has any error tag."""
        tags = self.s3_utils.get_tags(bucket_name, object_key)
        return any(tag in tags for tag in self.error_tags)

    def mark_document_processed(self, bucket_name: str, object_key: str) -> None:
        """Mark a document as processed by adding a tag."""
        try:
            tag_dict = self.s3_utils.get_tags(bucket_name, object_key)
            tag_set = [{"Key": k, "Value": v} for k, v in tag_dict.items()]

            # Add processed tag
            tag_set.append({
                "Key": self.config.INGESTION.PROCESSED_TAG,
                "Value": datetime.now(UTC).isoformat(),
            })

            self.s3_utils.update_tags(bucket_name, object_key, tag_set)
            self.logger.debug(f"Marked {object_key} as processed")
        except Exception as e:
            self.logger.warning(f"Failed to mark {object_key} as processed", error=str(e))

    def add_error_tag(self, bucket_name: str, object_key: str, error_tag: str, error_message: str = "") -> None:
        """Add an error tag to a document."""
        try:
            existing_tags = self.s3_utils.get_tags(bucket_name, object_key)

            # Remove any existing processing tags
            tag_set = [{"Key": k, "Value": v} for k, v in existing_tags.items() if k not in self.all_processing_tags]

            # Add the new error tag
            tag_value = error_message if error_message else datetime.now(UTC).isoformat()
            tag_set.append({"Key": error_tag, "Value": tag_value})

            self.s3_utils.update_tags(bucket_name, object_key, tag_set)
            self.logger.info(f"Tagged {object_key} with error tag {error_tag}")
        except Exception as e:
            self.logger.warning(f"Failed to add error tag {error_tag} to {object_key}", error=str(e))

    def remove_processed_tag(self, bucket_name: str, object_key: str) -> None:
        """Remove the processed tag from a document."""
        try:
            existing_tags = self.s3_utils.get_tags(bucket_name, object_key)

            # Remove processed tag
            tag_set = [
                {"Key": k, "Value": v} for k, v in existing_tags.items() if k != self.config.INGESTION.PROCESSED_TAG
            ]

            self.s3_utils.update_tags(bucket_name, object_key, tag_set)
            self.logger.debug(f"Removed processed tag from {object_key}")
        except Exception as e:
            self.logger.warning(f"Failed to remove processed tag from {object_key}", error=str(e))


class S3FileClassifier:
    """Utility class for classifying and filtering S3 files."""

    @staticmethod
    def is_symlink_file(object_key: str) -> bool:
        """Check if the file appears to be a symlink based on the filename."""
        # Symlinks often have specific patterns or extensions
        # This is a heuristic approach since S3 doesn't preserve symlink metadata
        filename = Path(object_key).name.lower()

        # Common symlink indicators
        symlink_indicators = [
            ".lnk",  # Windows shortcuts
            "desktop.ini",  # Windows desktop configuration
            ".url",  # Internet shortcuts
            ".symlink",  # Explicit symlink files
        ]

        return any(filename.endswith(indicator) for indicator in symlink_indicators)

    @staticmethod
    def is_temp_office_file(object_key: str) -> bool:
        """Check if the file is a temporary MS Office file."""
        filename = Path(object_key).name.lower()

        # Check for temporary file patterns
        return (
            filename.startswith("~$")
            or filename.endswith(".tmp")
            or any(pattern in filename for pattern in ["~wrd", "~xls", "~ppt"])
        )

    @staticmethod
    def is_supported_file_type(object_key: str, supported_formats: set[str]) -> bool:
        """Check if the file type is supported by the processing pipeline."""
        return any(object_key.lower().endswith(ext) for ext in supported_formats)
