from minio import Minio
from minio.error import S3Error
import os


class MinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

    def upload_file(self, bucket_name, folder_name, file_path):
        # Ensure bucket exists
        if not self.client.bucket_exists(bucket_name):
            print(f"Bucket '{bucket_name}' does not exist. Creating it.")
            self.client.make_bucket(bucket_name)
        else:
            print(f"Bucket '{bucket_name}' already exists.")

        # Ensure folder exists as a prefix (MinIO uses prefixes for folders)
        folder_path = folder_name.rstrip("/") + "/"
        object_name = os.path.join(folder_path, os.path.basename(file_path)).replace("\\", "/")

        # Upload the file
        self.client.fput_object(bucket_name, object_name, file_path)
        print(f"File '{file_path}' uploaded to '{folder_name}' in bucket '{bucket_name}' as '{object_name}'.")

