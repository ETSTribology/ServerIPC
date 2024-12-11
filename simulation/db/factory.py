import logging
from typing import Type, Dict, Any

from simulation.db.db import DatabaseBase
from simulation.db.postgres import Postgres
from simulation.db.mysql import MySQL
from simulation.core.utils.singleton import SingletonMeta


class DatabaseFactory(metaclass=SingletonMeta):
    """
    Factory for creating and managing database instances.
    """

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> DatabaseBase:
        """
        Create and return a backend instance based on the configuration.

        Args:
            config: A dictionary containing the database configuration.

        Returns:
            An instance of the database class.

        Raises:
            ValueError: If the database type is unknown.
        """
        logger = logging.getLogger("DatabaseFactory")
        logger.info("Creating database...")

        database_config = config.get("database", {})
        database_type = database_config.get("type", "mysql").lower()

        if database_type not in DatabaseFactory._instances:
            if database_type in ["postgres"]:
                database_instance = Postgres(database_config)
            elif database_type == "mysql":
                database_instance = MySQL(database_config)
            # elif database_type == "influxdb":
            #     database_instance = InfluxDB(database_config)
            else:
                raise ValueError(f"Unknown database type: {database_type}")

            DatabaseFactory._instances[database_type] = database_instance

        return DatabaseFactory._instances[database_type]
