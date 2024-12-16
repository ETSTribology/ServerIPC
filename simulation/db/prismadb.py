from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Union

from prisma import Prisma
from prisma.errors import PrismaError

from simulation.db.db import DatabaseBase
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)


def handle_prisma_errors(func):
    """Decorator to handle Prisma errors and log them."""
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            return func(*args, **kwargs)
        except PrismaError as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"PrismaError in {func.__name__}: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.DATABASE_OPERATION,
                f"PrismaError in {func.__name__}",
                details=str(e),
            )
        except Exception as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unexpected error in {func.__name__}: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.DATABASE_OPERATION,
                f"Unexpected error in {func.__name__}",
                details=str(e),
            )
    return wrapper


class PrismaDB(DatabaseBase):
    """Prisma implementation of DatabaseBase."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Prisma(auto_register=True)
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details("PrismaDB instance created.")
        )

    def connect_client(self):
        try:
            self.client.connect()
            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    "Prisma client connected successfully."
                )
            )
        except PrismaError as e:
            self.logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Failed to connect Prisma client: {e}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.DATABASE_CONNECTION,
                "Failed to connect Prisma client",
                details=str(e),
            )

    def close(self):
        if self.client.is_connected():
            self.client.disconnect()
            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details("Prisma client disconnected.")
            )

    @handle_prisma_errors
    def create(self, table: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record in a given table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        result = model.create(data=record)
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Record created in table '{table}': {result}"
            )
        )
        return result

    @handle_prisma_errors
    def read(self, table: str, record_id: str) -> Union[Dict[str, Any], None]:
        """Read a record by ID from a given table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        result = model.find_unique(where={"id": record_id})
        if result:
            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Record retrieved from table '{table}': {result}"
                )
            )
        else:
            self.logger.warning(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"No record found with ID '{record_id}' in table '{table}'."
                )
            )
        return result

    @handle_prisma_errors
    def update(self, table: str, record_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update a record by ID in a given table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        result = model.update(where={"id": record_id}, data=updates)
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Record '{record_id}' updated in table '{table}': {result}"
            )
        )
        return result

    @handle_prisma_errors
    def delete(self, table: str, record_id: str) -> None:
        """Delete a record by ID from a given table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        model.delete(where={"id": record_id})
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Record '{record_id}' deleted from table '{table}'."
            )
        )

    @handle_prisma_errors
    def list_all(self, table: str) -> List[Dict[str, Any]]:
        """List all records in a given table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        results = model.find_many()
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Retrieved all records from table '{table}': {results}"
            )
        )
        return results

    @handle_prisma_errors
    def raw_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a raw SQL query."""
        results = self.client.execute_raw(query)
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Raw query executed: {query}"
            )
        )
        return results

    # Schema-related methods
    @handle_prisma_errors
    def create_table(self, table: str, schema: Dict[str, Any]) -> None:
        """Create a new table with the specified schema."""
        raise NotImplementedError("Prisma does not support dynamic table creation.")

    @handle_prisma_errors
    def delete_table(self, table: str) -> None:
        """Delete the specified table."""
        raise NotImplementedError("Prisma does not support dynamic table deletion.")

    @handle_prisma_errors
    def create_all_tables(self) -> None:
        """Create all tables in the schema."""
        raise NotImplementedError("Prisma does not support dynamic schema creation.")

    @handle_prisma_errors
    def delete_all_tables(self) -> None:
        """Delete all tables in the schema."""
        raise NotImplementedError("Prisma does not support dynamic schema deletion.")

    @handle_prisma_errors
    def get_tables(self) -> List[str]:
        """List all tables in the database."""
        for model in self.client.models:
            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Table found in Prisma schema: {model}"
                )
            )
        return self.client.models

    @handle_prisma_errors
    def get_schema(self, table: str) -> Dict[str, Any]:
        """Retrieve the schema of the specified table."""
        self.logger.info(
            f"Returning a placeholder schema for table '{table}'."
        )
        return {"table": table, "schema": "Placeholder schema."}

    @handle_prisma_errors
    def get(self, table: str, record_id: str) -> Union[Dict[str, Any], None]:
        """Retrieve a record by its ID from the specified table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        result = model.find_unique(where={"id": record_id})
        if result:
            self.logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    f"Record retrieved from table '{table}': {result}"
                )
            )
        else:
            self.logger.warning(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"No record found with ID '{record_id}' in table '{table}'."
                )
            )
        return result

    @handle_prisma_errors
    def get_table(self, table: str) -> List[Dict[str, Any]]:
        """Retrieve all records from the specified table."""
        model = getattr(self.client, table.lower(), None)
        if not model:
            raise SimulationError(
                SimulationErrorCode.INVALID_OPERATION,
                f"Table '{table}' does not exist in the Prisma schema."
            )
        results = model.find_many()
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Retrieved all records from table '{table}': {results}"
            )
        )
        return results

    @handle_prisma_errors
    def query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SurrealQL or SQL-like query."""
        results = self.client.execute_raw(query)
        self.logger.info(
            SimulationLogMessageCode.COMMAND_SUCCESS.details(
                f"Raw query executed: {query}"
            )
        )
        return results
