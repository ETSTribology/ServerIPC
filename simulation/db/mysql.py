import json
import logging
from typing import List, Dict, Any

import mysql.connector
from mysql.connector.errors import Error

from simulation.db.db import DatabaseBase


class MySQL(DatabaseBase):
    """MySQL implementation of DatabaseBase."""

    def __init__(
        self,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.connection = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            self.logger.info("MySQL client initialized successfully.")
            self._ensure_table_exists("configurations")
        except Error as e:
            self.logger.error(f"Failed to initialize MySQL client: {e}")
            raise

    def _ensure_table_exists(self, table: str) -> None:
        """Ensure that the required table exists in the database."""
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                config JSON NOT NULL
            );
        """
        try:
            self.cursor.execute(create_table_query)
            self.logger.info(f"Table '{table}' ensured in MySQL.")
        except Error as e:
            self.logger.error(f"Failed to ensure table '{table}': {e}")
            raise

    def create_record(self, table: str, record: dict) -> dict:
        """Create a new record in the specified table."""
        insert_query = f"""
            INSERT INTO {table} (config) VALUES (%s)
        """
        try:
            config_json = json.dumps(record)
            self.cursor.execute(insert_query, (config_json,))
            self.connection.commit()
            record_id = self.cursor.lastrowid
            self.logger.info(f"Record created in table '{table}' with ID {record_id}.")
            return {"id": record_id, "config": record}
        except Error as e:
            self.logger.error(f"Failed to create record in table '{table}': {e}")
            raise

    def get_record(self, table: str, record_id: str) -> dict:
        """Retrieve a record by its ID from the specified table."""
        select_query = f"""
            SELECT id, config FROM {table} WHERE id = %s
        """
        try:
            self.cursor.execute(select_query, (record_id,))
            result = self.cursor.fetchone()
            if result:
                record_id, config_json = result
                config = json.loads(config_json)
                self.logger.info(f"Record retrieved from table '{table}': {result}")
                return {"id": record_id, "config": config}
            self.logger.warning(f"No record found with ID '{record_id}' in table '{table}'.")
            return {}
        except Error as e:
            self.logger.error(f"Failed to retrieve record '{record_id}' from table '{table}': {e}")
            raise

    def update_record(self, table: str, record_id: str, updates: dict) -> dict:
        """Update a record by its ID in the specified table."""
        try:
            # Fetch existing config
            select_query = f"SELECT config FROM {table} WHERE id = %s"
            self.cursor.execute(select_query, (record_id,))
            result = self.cursor.fetchone()
            if not result:
                self.logger.warning(f"No record found with ID '{record_id}' in table '{table}'.")
                return {}
            current_config = json.loads(result[0])
            # Update config
            current_config.update(updates)
            updated_config_json = json.dumps(current_config)
            update_query = f"""
                UPDATE {table}
                SET config = %s
                WHERE id = %s
            """
            self.cursor.execute(update_query, (updated_config_json, record_id))
            self.connection.commit()
            self.logger.info(f"Record '{record_id}' updated in table '{table}'.")
            return {"id": record_id, "config": current_config}
        except Error as e:
            self.logger.error(f"Failed to update record '{record_id}' in table '{table}': {e}")
            raise

    def delete_record(self, table: str, record_id: str) -> None:
        """Delete a record by its ID from the specified table."""
        delete_query = f"""
            DELETE FROM {table} WHERE id = %s
        """
        try:
            self.cursor.execute(delete_query, (record_id,))
            self.connection.commit()
            self.logger.info(f"Record '{record_id}' deleted from table '{table}'.")
        except Error as e:
            self.logger.error(f"Failed to delete record '{record_id}' from table '{table}': {e}")
            raise

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query."""
        try:
            self.cursor.execute(query)
            if self.cursor.with_rows:  # If the query returns rows
                columns = [desc[0] for desc in self.cursor.description]
                results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
                self.logger.info(f"Query executed successfully: {query}")
                return results
            else:
                self.connection.commit()
                self.logger.info(f"Query executed successfully: {query}")
                return []
        except Error as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            raise
