import logging
import os
import subprocess
from typing import Any, Dict, Optional

from simulation.db.db import DatabaseBase
from simulation.db.prismadb import PrismaDB
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode

logger = logging.getLogger(__name__)

class DatabaseFactory:
    """Factory for creating and managing database instances."""

    _instances: Dict[str, DatabaseBase] = {}
    DEFAULT_DB_URL = "postgresql://user:password@localhost:5432/db"
    DEFAULT_PROVIDER = "postgresql"

    @staticmethod
    def _build_database_url(config: Dict[str, Any]) -> str:
        """Build database URL from config."""
        database_type = config.get("backend", "postgresql")
        database_config = config.get("config", {})

        if database_type == "sqlite":
            db_name = database_config.get("name", "database.db")
            return f"file:{db_name}"

        # PostgreSQL URL construction
        host = database_config.get("host", "localhost")
        port = database_config.get("port", 5432)
        name = database_config.get("name", "db")
        user = database_config.get("user", "user")
        password = database_config.get("password", "password")

        return f"{database_type}://{user}:{password}@{host}:{port}/{name}?schema=public"

    @staticmethod
    def _validate_env_vars(database_url: str, db_provider: str) -> None:
        """Validate environment variables."""
        if not database_url:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details("DATABASE_URL is required")
            )
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, "DATABASE_URL is required")
        if not db_provider:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("DB_PROVIDER is required"))
            raise SimulationError(SimulationErrorCode.INPUT_VALIDATION, "DB_PROVIDER is required")
        if db_provider not in ["postgresql", "sqlite"]:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Unsupported database provider: {db_provider}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.INPUT_VALIDATION,
                f"Unsupported database provider: {db_provider}",
            )

    @staticmethod
    def _initialize_prisma(database_url: str, db_provider: str) -> None:
        """Initialize Prisma with database configuration."""
        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details("Initializing Prisma..."))

        # Load environment variables
        os.environ["DATABASE_URL"] = database_url
        os.environ["DB_PROVIDER"] = db_provider

        # Validate schema file
        schema_path = os.path.join(os.getcwd(), "simulation", "db", "schema.prisma")
        if not os.path.exists(schema_path):
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Prisma schema file not found at {schema_path}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.FILE_IO, f"Prisma schema file not found at {schema_path}"
            )

        # Update schema.prisma
        with open(schema_path, "r") as file:
            schema_content = file.read()

        # Replace environment variables with actual values
        schema_content_new = schema_content.replace(
            'provider = env("DB_PROVIDER")', f'provider = "{db_provider}"'
        )
        schema_content_new = schema_content_new.replace(
            'url = env("DATABASE_URL")', f'url = "{database_url}"'
        )

        # Write updated schema
        with open(schema_path, "w") as file:
            file.write(schema_content_new)
            logger.info(
                SimulationLogMessageCode.COMMAND_SUCCESS.details(
                    "Updated schema.prisma configuration"
                )
            )

        # Validate models exist
        if "model" not in schema_content_new:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    "No models defined in schema.prisma"
                )
            )
            raise SimulationError(
                SimulationErrorCode.INPUT_VALIDATION, "No models defined in schema.prisma"
            )

        # Run prisma db push with environment variables
        prisma_dir = os.path.join(os.getcwd(), "simulation", "db")
        env_vars = {**os.environ, "DATABASE_URL": database_url, "DB_PROVIDER": db_provider}

        try:
            subprocess.run(
                ["prisma", "db", "push"],
                capture_output=True,
                text=True,
                cwd=prisma_dir,
                env=env_vars,
                check=True,
            )
            logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details("Prisma initialized successfully"))
        except subprocess.CalledProcessError as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_FAILED.details(
                    f"Prisma db push failed: {e.stderr}"
                )
            )
            raise SimulationError(
                SimulationErrorCode.DATABASE_SETUP, f"Prisma db push failed: {e.stderr}"
            )

    @staticmethod
    def create(config: Dict[str, Any]) -> DatabaseBase:
        db_provider = config.get("db", {}).get("backend", "postgresql")

        # Return existing instance if already created
        if db_provider in DatabaseFactory._instances:
            logger.warning(
                f"Database instance for provider '{db_provider}' is already registered."
            )
            return DatabaseFactory._instances[db_provider]

        database_url = DatabaseFactory._build_database_url(config.get("db", {}))
        try:
            # Initialize the Prisma client
            DatabaseFactory._initialize_prisma(database_url, db_provider)

            # Create and register the database instance
            database_instance = PrismaDB(config.get("db", {}).get("config", {}))
            database_instance.connect_client()
            DatabaseFactory._instances[db_provider] = database_instance

            logger.info("Database instance created and registered successfully.")
            return database_instance
        except Exception as e:
            logger.error(f"Failed to create or initialize the database: {e}")
            raise SimulationError(SimulationErrorCode.DATABASE_INITIALIZATION, str(e))

    @staticmethod
    def get_current_db() -> Optional[DatabaseBase]:
        """Get the current database instance from the factory."""
        if not DatabaseFactory._instances:
            logger.warning("No database instance available.")
            return None
        
        db_instance = next(iter(DatabaseFactory._instances.values()))
        if db_instance is None:
            logger.warning("No valid database instance found.")
        return db_instance