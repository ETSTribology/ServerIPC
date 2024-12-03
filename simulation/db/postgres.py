import json
import logging
from typing import List, Dict, Any

import psycopg2
from psycopg2 import sql

from simulation.core.db.db import DatabaseBase

class Postgres(DatabaseBase):
    """PostgreSQL implementation of DatabaseBase."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgrespassword",
        database: str = "postgres",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.connection = psycopg2.connect(
                host=host, port=port, user=user, password=password, dbname=database
            )
            self.connection.autocommit = True
            self.cursor = self.connection.cursor()
            self.logger.info("PostgreSQL client initialized successfully.")
            self._ensure_table_exists("configurations")
        except psycopg2.Error as e:
            self.logger.error(f"Failed to initialize PostgreSQL client: {e}")
            raise

    def _ensure_table_exists(self, table: str) -> None:
        """Ensure that the required table exists in the database."""
        create_table_query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id SERIAL PRIMARY KEY,
                config JSONB NOT NULL
            );
        """
        ).format(table=sql.Identifier(table))
        try:
            self.cursor.execute(create_table_query)
            self.logger.info(f"Table '{table}' ensured in PostgreSQL.")
        except psycopg2.Error as e:
            self.logger.error(f"Failed to ensure table '{table}': {e}")
            raise

    def create(self, table: str, record: dict) -> dict:
        """Create a new record in the specified table."""
        insert_query = sql.SQL(
            """
            INSERT INTO {table} (config) VALUES (%s) RETURNING id, config;
        """
        ).format(table=sql.Identifier(table))
        try:
            self.cursor.execute(insert_query, [json.dumps(record)])
            result = self.cursor.fetchone()
            self.logger.info(f"Record created in table '{table}': {result}")
            return {"id": result[0], "config": result[1]}
        except psycopg2.Error as e:
            self.logger.error(f"Failed to create record in table '{table}': {e}")
            raise

    def get(self, table: str, record_id: str) -> dict:
        """Retrieve a record by its ID from the specified table."""
        select_query = sql.SQL(
            """
            SELECT id, config FROM {table} WHERE id = %s;
        """
        ).format(table=sql.Identifier(table))
        try:
            self.cursor.execute(select_query, [record_id])
            result = self.cursor.fetchone()
            if result:
                self.logger.info(f"Record retrieved from table '{table}': {result}")
                return {"id": result[0], "config": result[1]}
            self.logger.warning(f"No record found with ID '{record_id}' in table '{table}'.")
            return {}
        except psycopg2.Error as e:
            self.logger.error(f"Failed to retrieve record '{record_id}' from table '{table}': {e}")
            raise

    def update(self, table: str, record_id: str, updates: dict) -> dict:
        """Update a record by its ID in the specified table."""
        update_query = sql.SQL(
            """
            UPDATE {table}
            SET config = config || %s::jsonb
            WHERE id = %s
            RETURNING id, config;
        """
        ).format(table=sql.Identifier(table))
        try:
            self.cursor.execute(update_query, [json.dumps(updates), record_id])
            result = self.cursor.fetchone()
            if result:
                self.logger.info(f"Record '{record_id}' updated in table '{table}': {result}")
                return {"id": result[0], "config": result[1]}
            self.logger.warning(
                f"No record found with ID '{record_id}' to update in table '{table}'."
            )
            return {}
        except psycopg2.Error as e:
            self.logger.error(f"Failed to update record '{record_id}' in table '{table}': {e}")
            raise

    def delete(self, table: str, record_id: str) -> None:
        """Delete a record by its ID from the specified table."""
        delete_query = sql.SQL(
            """
            DELETE FROM {table} WHERE id = %s;
        """
        ).format(table=sql.Identifier(table))
        try:
            self.cursor.execute(delete_query, [record_id])
            self.logger.info(f"Record '{record_id}' deleted from table '{table}'.")
        except psycopg2.Error as e:
            self.logger.error(f"Failed to delete record '{record_id}' from table '{table}': {e}")
            raise

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SQL query."""
        try:
            self.cursor.execute(query)
            if self.cursor.description:  # If the query returns rows
                columns = [desc[0] for desc in self.cursor.description]
                results = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
                self.logger.info(f"Query executed successfully: {query}")
                return results
            self.logger.info(f"Query executed successfully: {query}")
            return []
        except psycopg2.Error as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            raise