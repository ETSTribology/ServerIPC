import json
import logging
from typing import Any, Dict, List

from influxdb import InfluxDBClient, Point, WritePrecision
from influxdb.client.write_api import SYNCHRONOUS

from simulation.core.db.db import DatabaseBase


class InfluxDB(DatabaseBase):
    """InfluxDB implementation of DatabaseBase."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8086,
        token: str = "my-token",
        org: str = "my-org",
        bucket: str = "my-bucket",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.client = InfluxDBClient(
                url=f"http://{host}:{port}",
                token=token,
                org=org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.bucket = bucket
            self.org = org
            self.logger.info("InfluxDB client initialized successfully.")
            self._ensure_bucket_exists(bucket)
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")
            raise

    def _ensure_bucket_exists(self, bucket: str) -> None:
        """Ensure that the required bucket exists in InfluxDB."""
        try:
            buckets_api = self.client.buckets_api()
            existing_buckets = [b.name for b in buckets_api.find_buckets().buckets]
            if bucket not in existing_buckets:
                buckets_api.create_bucket(bucket=bucket, org=self.org)
                self.logger.info(f"Bucket '{bucket}' created in InfluxDB.")
            else:
                self.logger.info(f"Bucket '{bucket}' already exists in InfluxDB.")
        except Exception as e:
            self.logger.error(f"Failed to ensure bucket '{bucket}': {e}")
            raise

    def create(self, table: str, record: dict) -> dict:
        """
        Create a new record in the specified bucket (table).
        InfluxDB uses measurements instead of tables.
        """
        try:
            point = Point(table)
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    point = point.field(key, value)
                else:
                    point = point.tag(key, str(value))
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            self.logger.info(f"Record written to measurement '{table}': {record}")
            return record  # InfluxDB does not return an ID by default
        except Exception as e:
            self.logger.error(f"Failed to create record in measurement '{table}': {e}")
            raise

    def get(self, table: str, record_id: str) -> dict:
        """
        Retrieve records based on a unique identifier.
        Since InfluxDB is a time-series database, retrieval is done via queries.
        """
        try:
            query = f'''
            from(bucket:"{self.bucket}")
              |> range(start: -100y)  // Adjust the range as needed
              |> filter(fn: (r) => r._measurement == "{table}" and r.id == "{record_id}")
            '''
            result = self.query_api.query(org=self.org, query=query)
            records = []
            for table_result in result:
                for record in table_result.records:
                    records.append(record.values)
            self.logger.info(f"Records retrieved from measurement '{table}': {records}")
            return records[0] if records else {}
        except Exception as e:
            self.logger.error(f"Failed to retrieve record '{record_id}' from measurement '{table}': {e}")
            raise

    def update(self, table: str, record_id: str, updates: dict) -> dict:
        """
        Update a record by creating a new point with the same ID and updated fields.
        InfluxDB is append-only; to "update", you write a new point with the desired changes.
        """
        try:
            # Retrieve existing record
            existing_record = self.get(table, record_id)
            if not existing_record:
                self.logger.warning(f"No record found with ID '{record_id}' in measurement '{table}'.")
                return {}
            # Update fields
            existing_record.update(updates)
            # Write updated record
            self.create(table, existing_record)
            self.logger.info(f"Record '{record_id}' updated in measurement '{table}'.")
            return existing_record
        except Exception as e:
            self.logger.error(f"Failed to update record '{record_id}' in measurement '{table}': {e}")
            raise

    def delete(self, table: str, record_id: str) -> None:
        """
        Delete records by setting a range that matches the record's ID.
        Note: InfluxDB 2.x does not support per-record deletion directly.
        You need to use the Delete API with a predicate.
        """
        try:
            delete_api = self.client.delete_api()
            delete_api.delete(
                start="1970-01-01T00:00:00Z",
                stop="2099-12-31T23:59:59Z",
                predicate=f'_measurement="{table}" AND id="{record_id}"',
                bucket=self.bucket,
                org=self.org
            )
            self.logger.info(f"Record '{record_id}' deleted from measurement '{table}'.")
        except Exception as e:
            self.logger.error(f"Failed to delete record '{record_id}' from measurement '{table}': {e}")
            raise

    def query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a Flux query and return the results.
        """
        try:
            result = self.query_api.query(org=self.org, query=query)
            records = []
            for table_result in result:
                for record in table_result.records:
                    records.append(record.values)
            self.logger.info(f"Query executed successfully: {query}")
            return records
        except Exception as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            raise
