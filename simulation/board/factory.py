import logging
from typing import Any, Dict

from board.base import BoardBase
from board.factory_base import BoardFactoryBase
from board.grafana_board import GrafanaBoard

class BoardFactory(BoardFactoryBase):
    """
    Factory for creating and managing board instances.
    """

    _instances = {}

    def create_board(self, config: Dict[str, Any]) -> BoardBase:
        """
        Create and return a board backend instance based on the configuration.

        Args:
            config: Configuration dictionary for the board backend.

        Returns:
            An instance of a BoardBase subclass.

        Raises:
            ValueError: If the board type is unknown.
        """
        logger = logging.getLogger(self.__class__.__name__)
        board_type = config.get("type", "grafana").lower()

        if board_type in self._instances:
            logger.debug(f"Returning existing instance for board type '{board_type}'.")
            return self._instances[board_type]

        if board_type == "grafana":
            board_instance = GrafanaBoard(config)
        else:
            raise ValueError(f"Unknown board type: {board_type}")

        self._instances[board_type] = board_instance
        logger.info(f"Created new board instance for type '{board_type}'.")
        return board_instance
