import logging
from functools import lru_cache
from typing import Type

from simulation.core.db.db import DatabaseBase
from simulation.core.utils.singleton import SingletonMeta

class DatabaseFactory(metaclass=SingletonMeta):
    """Factory for creating database instances."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @lru_cache(maxsize=None)
    def get_class(self, type_lower: str) -> Type[DatabaseBase]:
        """Retrieve and cache the database class from the registry."""
        db_cls = self.registry_container.get_database_class(type_lower)
        if not db_cls:
            self.logger.error(f"No database class registered under name '{type_lower}'.")
            raise ValueError(f"No database class registered under name '{type_lower}'.")
        return db_cls

    def create(self, type: str, **kwargs) -> DatabaseBase:
        """Factory method to create a database instance.

        :param type: The type of database (e.g., 'sqlite').
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
            raise RuntimeError(f"Error during database initialization for type '{type_lower}': {e}")
