#!/usr/bin/env python3
"""CLI utility to upload documents from local directories to S3 buckets."""

import argparse
import sys
from pathlib import Path

from rag.connectors.docling_loader import DoclingAPILoader
from rag.utils.config import AppConfig, ConfigurationManager
from rag.utils.logger import get_logger
from rag.utils.s3 import S3Utils


class S3DocumentUploader:
    """Utility for uploading documents to S3 buckets."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Initialize the S3 document uploader."""
        self.config: AppConfig = config or ConfigurationManager.get_config()
        self.logger = get_logger()

        # S3 utilities setup
        self.s3_utils = S3Utils(config)

    def upload_directory(self, local_dir: Path, access_role: str, preserve_structure: bool = True) -> None:
        """Upload all supported files from a directory to the appropriate S3 bucket."""
        bucket_name = self.s3_utils.get_bucket_name()
        self.s3_utils.ensure_bucket_exists(bucket_name)

        # Find all supported files
        supported_files = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in DoclingAPILoader.SUPPORTED_FORMATS:
                supported_files.append(file_path)

        total_files = len(supported_files)
        self.logger.info(f"Found {total_files} supported files in {local_dir}")

        success_count = 0
        for i, file_path in enumerate(supported_files):
            self.logger.info(f"Uploading file {i + 1}/{total_files}: {file_path}")

            # Generate object key
            if preserve_structure:
                # Keep the relative path structure
                relative_path = file_path.relative_to(local_dir)
                object_key = str(relative_path).replace("\\", "/")  # Ensure forward slashes
            else:
                # Just use the filename
                object_key = file_path.name

            # Normalize the object key to make it machine-readable
            object_key = S3Utils.normalize_path(object_key)

            # Prefix with role directory (uppercased)
            object_key = f"{access_role.upper()}/{object_key}"

            if self.s3_utils.upload_file(file_path, bucket_name, object_key):
                success_count += 1

        self.logger.info(
            f"Upload completed: {success_count}/{total_files} files successfully uploaded to {bucket_name}"
        )

    def upload_data_directory(self, data_dir: Path, preserve_structure: bool = True) -> None:
        """Upload all subdirectories from the data directory to their respective S3 buckets."""
        if not data_dir.exists():
            self.logger.error(f"Data directory {data_dir} does not exist")
            return

        # Discover access roles from subdirectories
        access_roles = []
        for item in data_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                access_roles.append(item.name)

        if not access_roles:
            self.logger.warning(f"No subdirectories found in {data_dir}")
            return

        self.logger.info(f"Discovered access roles: {access_roles}")

        for role in access_roles:
            role_dir = data_dir / role
            self.logger.info(f"Processing access role '{role}' from directory {role_dir}")
            self.upload_directory(role_dir, role, preserve_structure)


def main() -> None:
    """Main entry point for the S3 upload utility."""
    parser = argparse.ArgumentParser(description="Upload documents to S3 buckets for ingestion")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: upload-all (uploads entire data directory with all subdirectories)
    upload_all_parser = subparsers.add_parser(
        "upload-all", help="Upload all subdirectories from the data directory to their respective S3 buckets"
    )
    _ = upload_all_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Data directory containing subdirectories for each access role (default: data)",
    )
    _ = upload_all_parser.add_argument(
        "--flat", action="store_true", help="Upload files without preserving directory structure (flatten)"
    )

    # Command: upload-role (uploads specific access role directory)
    upload_role_parser = subparsers.add_parser("upload-role", help="Upload documents for a specific access role")
    _ = upload_role_parser.add_argument("source_dir", type=Path, help="Source directory containing documents to upload")
    _ = upload_role_parser.add_argument("access_role", help="Access role (determines target bucket), e.g., EL, SH, EL2")
    _ = upload_role_parser.add_argument(
        "--flat", action="store_true", help="Upload files without preserving directory structure (flatten)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logger = get_logger()

    try:
        uploader = S3DocumentUploader()

        if args.command == "upload-all":
            if not args.data_dir.exists():
                print(f"Error: Data directory {args.data_dir} does not exist")
                sys.exit(1)

            uploader.upload_data_directory(data_dir=args.data_dir, preserve_structure=not args.flat)

        elif args.command == "upload-role":
            if not args.source_dir.exists():
                print(f"Error: Source directory {args.source_dir} does not exist")
                sys.exit(1)

            if not args.source_dir.is_dir():
                print(f"Error: {args.source_dir} is not a directory")
                sys.exit(1)

            uploader.upload_directory(
                local_dir=args.source_dir, access_role=args.access_role, preserve_structure=not args.flat
            )

        logger.info("Upload process completed successfully")
    except KeyboardInterrupt:
        logger.info("Upload process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Upload process failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
