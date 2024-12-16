from abc import ABC, abstractmethod


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

    @abstractmethod
    def create_table(self, table: str, schema: dict) -> None:
        """Create a new table with the specified schema."""
        pass

    @abstractmethod
    def delete_table(self, table: str) -> None:
        """Delete the specified table."""
        pass

    @abstractmethod
    def create_all_tables(self) -> None:
        pass

    @abstractmethod
    def delete_all_tables(self) -> None:
        pass

    @abstractmethod
    def get_table(self, table: str) -> list:
        """Retrieve all records from the specified table."""
        pass

    @abstractmethod
    def get_tables(self) -> list:
        """List all tables in the database."""
        pass

    @abstractmethod
    def get_schema(self, table: str) -> dict:
        """Retrieve the schema of the specified table."""
        pass
