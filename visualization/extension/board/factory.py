import logging
from typing import Any, Dict

from visualization.core.utils.singleton import SingletonMeta
from visualization.extension.board.tensorboard import TensorBoard

logger = logging.getLogger(__name__)


class BoardFactory(metaclass=SingletonMeta):
    """Factory class for creating board loggers."""

    _instances = {}

    @staticmethod
    def create(config: Dict[str, Any]) -> Any:
        """
        Create and return a storage instance based on the configuration.

        Args:
            config: A dictionary containing the storage configuration.

        Returns:
            An instance of the storage class.

        Raises:
            ValueError: If the storage type is not recognized or required fields are missing.
        """
        logger.info("Creating board...")
        board_config = config.get("extensions", {}).get("board", {})
        board_type = board_config.get("name", "tensorboard")

        if board_type not in BoardFactory._instances:
            if board_type == "tensorboard":
                board_instance = TensorBoard(config)
            else:
                raise ValueError(f"Unknown board type: {board_type}")

            BoardFactory._instances[board_type] = board_instance

        return BoardFactory._instances[board_type]
