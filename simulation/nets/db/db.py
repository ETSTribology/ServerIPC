import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List

import psycopg2
from core.registry.container import RegistryContainer
from core.registry.decorators import register
from psycopg2 import sql
from surrealdb import SurrealDB as SurrealClient
from surrealdb import SurrealDbError


class DatabaseBase(ABC):
    """Abstract base class for database storage backends."""

    @abstractmethod
    def create_record(self, table: str, record: dict) -> dict:
        """Create a new record in the specified table."""
        pass

    @abstractmethod
    def get_record(self, table: str, record_id: str) -> dict:
        """Retrieve a record by its ID from the specified table."""
        pass

    @abstractmethod
    def update_record(self, table: str, record_id: str, updates: dict) -> dict:
        """Update a record by its ID in the specified table."""
        pass

    @abstractmethod
    def delete_record(self, table: str, record_id: str) -> None:
        """Delete a record by its ID from the specified table."""
        pass

    @abstractmethod
    def query(self, query: str) -> list:
        """Execute a SurrealQL or SQL-like query."""
        pass


@register(type="database", name="surrealdb")
class SurrealDBStorage(DatabaseBase):
    """SurrealDB storage implementation of DatabaseBase."""

    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        namespace: str = "test",
        database: str = "test",
        username: str = "root",
        password: str = "rootpassword",
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        try:
            self.db = SurrealClient(url)
            self.db.signin({"user": username, "pass": password})
            self.db.use(namespace, database)
            self.logger.info("SurrealDB client initialized successfully.")
        except SurrealDbError as e:
            self.logger.error(f"Failed to initialize SurrealDB client: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Unexpected error during SurrealDB client initialization: {e}"
            )
            raise

    def create_record(self, table: str, record: dict) -> dict:
        try:
            result = self.db.create(table, record).result()
            self.logger.info(f"Record created in table '{table}': {result}")
            return result
        except SurrealDbError as e:
            self.logger.error(f"Failed to create record in table '{table}': {e}")
            raise

    def get_record(self, table: str, record_id: str) -> dict:
        try:
            record_key = f"{table}:{record_id}"
            result = self.db.select(record_key).result()
            self.logger.info(f"Record retrieved from table '{table}': {result}")
            return result
        except SurrealDbError as e:
            self.logger.error(
                f"Failed to retrieve record '{record_id}' from table '{table}': {e}"
            )
            raise

    def update_record(self, table: str, record_id: str, updates: dict) -> dict:
        try:
            record_key = f"{table}:{record_id}"
            result = self.db.update(record_key, updates).result()
            self.logger.info(
                f"Record '{record_id}' updated in table '{table}': {result}"
            )
            return result
        except SurrealDbError as e:
            self.logger.error(
                f"Failed to update record '{record_id}' in table '{table}': {e}"
            )
            raise

    def delete_record(self, table: str, record_id: str) -> None:
        try:
            record_key = f"{table}:{record_id}"
            self.db.delete(record_key).result()
            self.logger.info(f"Record '{record_id}' deleted from table '{table}'.")
        except SurrealDbError as e:
            self.logger.error(
                f"Failed to delete record '{record_id}' from table '{table}': {e}"
            )
            raise

    def query(self, query: str) -> List[Dict[str, Any]]:
        try:
            result = self.db.query(query).result()
            self.logger.info(f"Query executed successfully: {query}")
            return result
        except SurrealDbError as e:
            self.logger.error(f"Failed to execute query '{query}': {e}")
            raise


@register(type="database", name="postgres")
class PostgresStorage(DatabaseBase):
    """PostgreSQL storage implementation of DatabaseBase."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgrespassword",
        database: str = "surreal_config",
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

    def create_record(self, table: str, record: dict) -> dict:
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

    def get_record(self, table: str, record_id: str) -> dict:
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
            self.logger.warning(
                f"No record found with ID '{record_id}' in table '{table}'."
            )
            return {}
        except psycopg2.Error as e:
            self.logger.error(
                f"Failed to retrieve record '{record_id}' from table '{table}': {e}"
            )
            raise

    def update_record(self, table: str, record_id: str, updates: dict) -> dict:
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
                self.logger.info(
                    f"Record '{record_id}' updated in table '{table}': {result}"
                )
                return {"id": result[0], "config": result[1]}
            self.logger.warning(
                f"No record found with ID '{record_id}' to update in table '{table}'."
            )
            return {}
        except psycopg2.Error as e:
            self.logger.error(
                f"Failed to update record '{record_id}' in table '{table}': {e}"
            )
            raise

    def delete_record(self, table: str, record_id: str) -> None:
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
            self.logger.error(
                f"Failed to delete record '{record_id}' from table '{table}': {e}"
            )
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


class DatabaseFactory:
    """Factory for creating database instances."""

    def __init__(self):
        self.registry_container = RegistryContainer()
        self.logger = logging.getLogger(self.__class__.__name__)

    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str) -> Type[DatabaseBase]:
        """Retrieve and cache the database class from the registry."""
        db_cls = self.registry_container.get_database_class(type_lower)
        if not db_cls:
            self.logger.error(
                f"No database class registered under name '{type_lower}'."
            )
            raise ValueError(f"No database class registered under name '{type_lower}'.")
        return db_cls

    def create(self, type: str, **kwargs) -> DatabaseBase:
        """Factory method to create a database instance.

        :param type: The type of database (e.g., 'surrealdb', 'postgres').
        :param kwargs: Additional parameters for the database backend.
        :return: An instance of DatabaseBase.
        """
        type_lower = type.lower()
        try:
            db_cls = self.get_class(type_lower)
            db_instance = db_cls(**kwargs)
            self.logger.info(f"Database '{type_lower}' created successfully.")
            return db_instance
        except Exception as e:
            self.logger.error(f"Failed to create database '{type_lower}': {e}")
            raise RuntimeError(
                f"Error during database initialization for type '{type_lower}': {e}"
            )
