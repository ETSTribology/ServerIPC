import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Dict, List, Type

import psycopg2
from psycopg2 import sql
from surrealdb import SurrealDB as SurrealClient
from surrealdb import SurrealDbError
from simulation.core.utils.singleton import SingletonMeta


class DatabaseBase(ABC):
    """Abstract base class for database storage backends."""

    @abstractmethod
    def create(self, table: str, record: dict) -> dict:
        """Create a new record in the specified table."""
        pass

    @abstractmethod
    def get(self, table: str, record_id: str) -> dict:
        """Retrieve a record by its ID from the specified table."""
        pass

    @abstractmethod
    def update(self, table: str, record_id: str, updates: dict) -> dict:
        """Update a record by its ID in the specified table."""
        pass

    @abstractmethod
    def delete(self, table: str, record_id: str) -> None:
        """Delete a record by its ID from the specified table."""
        pass

    @abstractmethod
    def query(self, query: str) -> list:
        """Execute a SurrealQL or SQL-like query."""
        pass