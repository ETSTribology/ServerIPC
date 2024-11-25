from minio import Minio
from minio.error import S3Error
import os
import logging

logger = logging.getLogger(__name__)

class MinioClient:
    """
    MinioClient handles file uploads to MinIO storage.
    """
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        logger.info(f"Initialized MinioClient with endpoint={endpoint}")

    def upload_file(self, bucket_name, folder_name, file_path):
        # Ensure bucket exists
        if not self.client.bucket_exists(bucket_name):
            logger.info(f"Bucket '{bucket_name}' does not exist. Creating it.")
            try:
                self.client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created.")
            except S3Error as e:
                logger.error(f"Failed to create bucket '{bucket_name}': {e}")
                raise
        else:
            logger.info(f"Bucket '{bucket_name}' already exists.")

        # Ensure folder exists as a prefix (MinIO uses prefixes for folders)
        folder_path = folder_name.rstrip("/") + "/"
        object_name = os.path.join(folder_path, os.path.basename(file_path)).replace("\\", "/")

        # Upload the file
        try:
            self.client.fput_object(bucket_name, object_name, file_path)
            logger.info(f"File '{file_path}' uploaded to '{folder_name}' in bucket '{bucket_name}' as '{object_name}'.")
        except S3Error as e:
            logger.error(f"Failed to upload file '{file_path}' to MinIO: {e}")
            raise
