import abc
import logging
import os
from typing import Any, Dict, Optional
from functools import lru_cache

import torch
import hydra
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register
from simulation.core.config import get_config

logger = logging.getLogger(__name__)


class BoardBase(abc.ABC):
    """Abstract base class for board loggers."""

    @abc.abstractmethod
    def log_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        """Logs a scalar value."""
        pass

    @abc.abstractmethod
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
    ) -> None:
        """Logs multiple scalar values."""
        pass

    @abc.abstractmethod
    def log_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: Optional[str] = "tensorflow",
    ) -> None:
        """Logs a histogram of values."""
        pass

    @abc.abstractmethod
    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        dataformats: str = "CHW",
    ) -> None:
        """Logs an image."""
        pass

    @abc.abstractmethod
    def log_figure(
        self,
        tag: str,
        figure,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> None:
        """Logs a matplotlib figure."""
        pass

    @abc.abstractmethod
    def log_text(
        self, tag: str, text_string: str, global_step: Optional[int] = None
    ) -> None:
        """Logs a text string."""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """Flushes any pending logs."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger and releases resources."""
        pass

    def __enter__(self):
        """Enables usage of the class with the 'with' statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensures that resources are cleaned up when exiting a 'with' block."""
        self.close()


registry_container = RegistryContainer()
registry_container.add_registry("board", "simulation.board.board.BoardBase")


@register(type="board", name="tensorboard")
class TensorBoardLogger(BoardBase):
    """A singleton logger that writes logs to TensorBoard and saves tensor data in a new directory."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        **kwargs
    ):
        """
        Initializes the TensorBoard logger with Hydra configuration support.

        Args:
            cfg (DictConfig, optional): Hydra configuration dictionary
            Additional arguments can override Hydra configuration
        """
        # Check if the instance has already been initialized to prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Get board configuration from cfg or from global config
        if cfg is None:
            cfg = get_config().board

        # Extract extra_config for TensorBoard specific settings
        tensorboard_cfg = cfg.get('extra_config', {})

        # Override configuration with explicit arguments or use defaults
        self.log_dir = kwargs.get('log_dir', tensorboard_cfg.get('log_dir', './runs'))
        tensor_data_subdir = kwargs.get('tensor_data_subdir', tensorboard_cfg.get('tensor_data_subdir', 'tensor_data'))
        comment = kwargs.get('comment', tensorboard_cfg.get('comment', ''))
        purge_step = kwargs.get('purge_step', tensorboard_cfg.get('purge_step', None))
        max_queue = kwargs.get('max_queue', tensorboard_cfg.get('max_queue', 10))
        flush_secs = kwargs.get('flush_secs', tensorboard_cfg.get('flush_secs', 120))
        filename_suffix = kwargs.get('filename_suffix', tensorboard_cfg.get('filename_suffix', ''))

        self.tensor_data_dir = os.path.join(self.log_dir, tensor_data_subdir)
        os.makedirs(self.tensor_data_dir, exist_ok=True)

        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        logger.info(f"TensorBoard logger initialized at '{self.log_dir}'.")
        logger.info(f"Tensor data will be saved in '{self.tensor_data_dir}'.")

        self._initialized = True  # Mark as initialized

    def log_scalar(
        self, tag: str, scalar_value: float, global_step: Optional[int] = None
    ) -> None:
        """Logs a scalar value."""
        self.writer.add_scalar(tag, scalar_value, global_step)
        logger.debug(f"Logged scalar '{tag}': {scalar_value} at step {global_step}.")

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        global_step: Optional[int] = None,
    ) -> None:
        """Logs multiple scalar values."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
        logger.debug(
            f"Logged scalars under main tag '{main_tag}' at step {global_step}."
        )

    def log_histogram(
        self,
        tag: str,
        values: Any,
        global_step: Optional[int] = None,
        bins: Optional[str] = "tensorflow",
    ) -> None:
        """Logs a histogram of values and saves tensor data to a new directory."""
        self.writer.add_histogram(tag, values, global_step, bins)
        # Save tensor data to new directory
        tensor_data_path = os.path.join(
            self.tensor_data_dir, f"{tag}_{global_step}.pt"
        )
        torch.save(values, tensor_data_path)
        logger.debug(
            f"Logged histogram '{tag}' at step {global_step}. Tensor data saved to '{tensor_data_path}'."
        )

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: Optional[int] = None,
        dataformats: str = "CHW",
    ) -> None:
        """Logs an image and saves tensor data to a new directory."""
        self.writer.add_image(tag, img_tensor, global_step, dataformats)
        # Save tensor data to new directory
        tensor_data_path = os.path.join(
            self.tensor_data_dir, f"{tag}_{global_step}.pt"
        )
        torch.save(img_tensor, tensor_data_path)
        logger.debug(
            f"Logged image '{tag}' at step {global_step}. Tensor data saved to '{tensor_data_path}'."
        )

    def log_figure(
        self,
        tag: str,
        figure,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> None:
        """Logs a matplotlib figure and saves it to a new directory."""
        self.writer.add_figure(tag, figure, global_step, close, walltime)
        # Save figure data to new directory
        figure_path = os.path.join(self.tensor_data_dir, f"{tag}_{global_step}.png")
        figure.savefig(figure_path)
        logger.debug(
            f"Logged figure '{tag}' at step {global_step}. Figure saved to '{figure_path}'."
        )

    def log_text(
        self, tag: str, text_string: str, global_step: Optional[int] = None
    ) -> None:
        """Logs a text string."""
        self.writer.add_text(tag, text_string, global_step)
        logger.debug(f"Logged text under tag '{tag}' at step {global_step}.")

    def flush(self) -> None:
        """Flushes the event file to disk."""
        self.writer.flush()
        logger.debug("Flushed TensorBoard events to disk.")

    def close(self) -> None:
        """Closes the TensorBoard writer."""
        self.writer.close()
        logger.info("TensorBoard logger closed.")


class BoardFactory:
    """Factory class for creating board loggers."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Board class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.board.get(type_lower)

    def create(self, type: Optional[str] = None, **kwargs) -> BoardBase:
        """
        Create a board logger with optional Hydra configuration.

        Args:
            type (str, optional): Type of board logger. 
                                  If not provided, uses default from configuration.
            **kwargs: Additional configuration parameters
        """
        try:
            # Get configuration
            cfg = get_config()
            
            # Determine board type
            type = type or cfg.board.type
            type_lower = type.lower()
            
            # Check if board is enabled
            if not cfg.board.get('enabled', True):
                logger.warning("Board logging is disabled in configuration.")
                return None

            # Set logging level
            log_level = cfg.board.get('log_level', 'INFO')
            logging.getLogger().setLevel(getattr(logging, log_level.upper()))

            # Retrieve board class and create instance
            board_cls = self.get_class(type_lower)
            return board_cls(cfg=cfg.board, **kwargs)

        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create board '{type}': {e}")
            raise