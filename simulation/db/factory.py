import asyncio
import logging
import os
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

from simulation.core.utils.singleton import SingletonMeta
from simulation.db.db import DatabaseBase
from simulation.db.prisma import PrismaDB
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


class DatabaseFactory(metaclass=SingletonMeta):
    """Factory for creating and managing database instances."""

    _instances: Dict[str, DatabaseBase] = {}
    DEFAULT_DB_URL = "postgresql://user:password@localhost:5432/db"
    DEFAULT_PROVIDER = "postgresql"

    @staticmethod
    def _build_database_url(config: Dict[str, Any]) -> Tuple[str, str]:
        """Build database URL from config."""
        database_type = config.get("backend", "postgresql")
        database_config = config.get("config", {})

        if database_type == "sqlite":
            db_name = database_config.get("name", "database.db")
            return f"file:{db_name}", database_type

        # PostgreSQL URL construction
        host = database_config.get("host", "localhost")
        port = database_config.get("port", 5432)
        name = database_config.get("name", "db")
        user = database_config.get("user", "user")
        password = database_config.get("password", "password")

        return (
            f"{database_type}://{user}:{password}@{host}:{port}/{name}?schema=public",
            database_type,
        )

    @staticmethod
    def _validate_env_vars(database_url: str, db_provider: str) -> None:
        """Validate environment variables."""
        if not database_url:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("DATABASE_URL is required"))
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, "DATABASE_URL is required")
        if not db_provider:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("DB_PROVIDER is required"))
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, "DB_PROVIDER is required")
        if db_provider not in ["postgresql", "sqlite"]:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unsupported database provider: {db_provider}"))
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, f"Unsupported database provider: {db_provider}")

    @staticmethod
    def _write_env_file(database_url: str, db_provider: str) -> None:
        """Write environment variables to .env file."""
        env_path = os.path.join(os.getcwd(), ".env")
        with open(env_path, "w") as file:
            file.write(f"DATABASE_URL={database_url}\n")
            file.write(f"DB_PROVIDER={db_provider}\n")
        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Environment variables written to .env file"))

    @staticmethod
    def _initialize_prisma(
        database_url: Optional[str] = None, db_provider: str = "postgresql"
    ) -> None:
        """Initialize Prisma with database configuration."""
        logger = logging.getLogger("DatabaseFactory")
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Initializing Prisma..."))

        # Load environment variables
        load_dotenv()

        # Use provided URL or fallback to environment/default
        database_url = database_url or os.getenv("DATABASE_URL", DatabaseFactory.DEFAULT_DB_URL)
        db_provider = db_provider or os.getenv("DB_PROVIDER", DatabaseFactory.DEFAULT_PROVIDER)

        # Validate environment variables
        DatabaseFactory._validate_env_vars(database_url, db_provider)

        # Write environment variables to .env file
        DatabaseFactory._write_env_file(database_url, db_provider)

        # Set environment variables
        os.environ["DATABASE_URL"] = database_url
        os.environ["DB_PROVIDER"] = db_provider

        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Using database provider: {db_provider}"))
        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Using database URL: {database_url}"))

        # Validate schema file
        schema_path = os.path.join(os.getcwd(), "simulation", "db", "schema.prisma")
        if not os.path.exists(schema_path):
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Prisma schema file not found at {schema_path}"))
            raise SimulationError(SimulationErrorCode.FILE_IO, f"Prisma schema file not found at {schema_path}")

        # Update schema.prisma
        with open(schema_path, "r") as file:
            schema_content = file.read()

        # Replace environment variables with actual values
        schema_content_new = re.sub(
            r'provider\s*=\s*env\(["\']DB_PROVIDER["\']\)',
            f'provider = "{db_provider}"',
            schema_content,
        )
        schema_content_new = re.sub(
            r'url\s*=\s*env\(["\']DATABASE_URL["\']\)',
            f'url = "{database_url}"',
            schema_content_new,
        )

        # Write updated schema
        with open(schema_path, "w") as file:
            file.write(schema_content_new)
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Updated schema.prisma configuration"))

        # Validate models exist
        if not re.search(r"model\s+\w+\s*{", schema_content_new):
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("No models defined in schema.prisma"))
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, "No models defined in schema.prisma")

        # Run prisma db push with environment variables
        prisma_dir = os.path.join(os.getcwd(), "simulation", "db")
        env_vars = {**os.environ, "DATABASE_URL": database_url, "DB_PROVIDER": db_provider}

        try:
            result = subprocess.run(
                ["prisma", "db", "push"],
                capture_output=True,
                text=True,
                cwd=prisma_dir,
                env=env_vars,
                check=True,
            )
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Prisma initialized successfully"))
        except subprocess.CalledProcessError as e:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Prisma db push failed: {e.stderr}"))
            raise SimulationError(SimulationErrorCode.DATABASE_SETUP, f"Prisma db push failed: {e.stderr}")

    @staticmethod
    async def create_async(config: Dict[str, Any]) -> DatabaseBase:
        """Asynchronous method to create database instance from config."""
        logger = logging.getLogger("DatabaseFactory")
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Creating database..."))

        # Build database URL from config
        database_url, database_type = DatabaseFactory._build_database_url(config.get("db", {}))

        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(f"Initializing {database_type} database with URL: {database_url}"))

        if database_type not in DatabaseFactory._instances:
            if database_type in ["postgresql", "sqlite"]:
                DatabaseFactory._initialize_prisma(
                    database_url=database_url, db_provider=database_type
                )
                database_instance = PrismaDB(config.get("db", {}).get("config", {}))

                # Connect the Prisma client before performing operations
                await database_instance.connect_client()

                # Removed manual table creation
                # await database_instance.create_all_tables()
            else:
                logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Unsupported database type: {database_type}"))
                raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, f"Unsupported database type: {database_type}")

            DatabaseFactory._instances[database_type] = database_instance

        return DatabaseFactory._instances[database_type]

    @staticmethod
    def create(config: Dict[str, Any]) -> DatabaseBase:
        """Create database instance from config."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop; create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(DatabaseFactory.create_async(config))