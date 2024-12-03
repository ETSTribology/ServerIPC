import logging
from typing import Type, Dict, Any

from simulation.core.db.db import DatabaseBase
from simulation.core.db.postgresql import PostgreSQL
from simulation.core.db.sqlite import SQLite
from simulation.core.utils.singleton import SingletonMeta

class DatabaseFactory(metaclass=SingletonMeta):
    """
    Factory for creating and managing database instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a backend instance based on the configuration.

        Args:
            config: A dictionary containing the database configuration.

        Returns:
            An instance of the database class.

        Raises:
            ValueError: 
        """
        logger.info("Creating database...")
        database_config = config.get("database", {})
        database_type = database_config.get("type", "sqlite").lower()

        if database_type not in DatabaseFactory._instances:
            if database_type == "sqlite":
                database_instance = SQLite(config)
            elif database_type == "postgresql":
                database_instance = PostgreSQL(config)
            else:
                raise ValueError(f"Unknown database type: {database_type}")

            DatabaseFactory._instances[database_type] = database_instance

        return DatabaseFactory._instances[database_type]
