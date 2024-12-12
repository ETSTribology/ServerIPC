import logging
from typing import List, Dict, Any, Callable
from functools import wraps
import asyncio

from prisma import Prisma
from prisma.errors import PrismaError

from simulation.db.db import DatabaseBase
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)

def handle_prisma_errors(func: Callable) -> Callable:
    """Decorator to handle Prisma errors and log them."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        self = args[0]
        try:
            return await func(*args, **kwargs)
        except PrismaError as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"PrismaError in {func.__name__}: {e}"))
            raise SimulationError(SimulationErrorCode.DATABASE_OPERATION, f"PrismaError in {func.__name__}", details=str(e))
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error in {func.__name__}: {e}"))
            raise SimulationError(SimulationErrorCode.DATABASE_OPERATION, f"Unexpected error in {func.__name__}", details=str(e))
    return wrapper

class PrismaDB(DatabaseBase):
    """Prisma implementation of DatabaseBase."""

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = Prisma()
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("PrismaDB instance created."))

    async def connect_client(self):
        try:
            await self.client.connect()
            self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Prisma client connected successfully."))
        except PrismaError as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Failed to connect Prisma client: {e}"))
            raise SimulationError(SimulationErrorCode.DATABASE_CONNECTION, "Failed to connect Prisma client", details=str(e))
        except Exception as e:
            self.logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unexpected error during client connection: {e}"))
            raise SimulationError(SimulationErrorCode.DATABASE_CONNECTION, "Unexpected error during client connection", details=str(e))

    @handle_prisma_errors
    async def create(self, table: str, record: dict) -> dict:
        result = await self.client[table].create(data=record)
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Record created in table '{table}': {result}"))
        return result

    @handle_prisma_errors
    async def get(self, table: str, record_id: str) -> dict:
        result = await self.client[table].find_unique(where={"id": record_id})
        if result:
            self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Record retrieved from table '{table}': {result}"))
            return result
        self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details(f"No record found with ID '{record_id}' in table '{table}'"))
        return {}

    @handle_prisma_errors
    async def update(self, table: str, record_id: str, updates: dict) -> dict:
        result = await self.client[table].update(where={"id": record_id}, data=updates)
        if result:
            self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Record '{record_id}' updated in table '{table}': {result}"))
            return result
        self.logger.warning(SimulationLogMessageCode.COMMAND_FAILED.details(f"No record found with ID '{record_id}' to update in table '{table}'"))
        return {}

    @handle_prisma_errors
    async def delete(self, table: str, record_id: str) -> None:
        await self.client[table].delete(where={"id": record_id})
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Record '{record_id}' deleted from table '{table}'"))

    @handle_prisma_errors
    async def query(self, query: str) -> List[Dict[str, Any]]:
        result = await self.client.execute_raw(query)
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Query executed successfully: {query}"))
        return result

    @handle_prisma_errors
    async def get_table(self, table: str) -> List[Dict[str, Any]]:
        result = await self.client[table].find_many()
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Records retrieved from table '{table}': {result}"))
        return result

    @handle_prisma_errors
    async def get_tables(self) -> List[str]:
        result = await self.client.execute_raw(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        tables = [row['table_name'] for row in result]
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Tables in the database: {tables}"))
        return tables

    @handle_prisma_errors
    async def get_schema(self, table: str) -> Dict[str, Any]:
        query = (
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{table}'"
        )
        result = await self.client.execute_raw(query)
        schema = {row['column_name']: row['data_type'] for row in result}
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Schema of table '{table}': {schema}"))
        return schema

    # Implementing abstract table management methods as no-ops
    @handle_prisma_errors
    async def create_table(self, table: str, schema: dict) -> None:
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Table creation is managed by Prisma. Skipping manual creation."))
        pass

    @handle_prisma_errors
    async def delete_table(self, table: str) -> None:
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Table deletion is managed by Prisma. Skipping manual deletion."))
        pass

    @handle_prisma_errors
    async def create_all_tables(self) -> None:
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Table creation is managed by Prisma. Skipping manual creation of all tables."))
        pass

    @handle_prisma_errors
    async def delete_all_tables(self) -> None:
        self.logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Table deletion is managed by Prisma. Skipping manual deletion of all tables."))
        pass