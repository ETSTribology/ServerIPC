import abc
import logging
import os
from typing import Any, Dict, Optional
from functools import lru_cache

import torch
from torch.utils.tensorboard import SummaryWriter
from simulation.core.utils.singleton import SingletonMeta

from simulation.core.registry.container import RegistryContainer
from simulation.core.registry.decorators import register

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
registry_container.add_registry("board", "board.board.BoardBase")


@register(type="board", name="tensorboard")
class TensorBoardLogger(BoardBase, metaclass=SingletonMeta):
    """A singleton logger that writes logs to TensorBoard and saves tensor data in a new directory."""

    def __init__(
        self,
        log_dir: str = "./runs",
        tensor_data_subdir: str = "tensor_data",
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        """
        Initializes the TensorBoard logger.

        Args:
            log_dir (str): Directory where TensorBoard logs will be saved.
            tensor_data_subdir (str): Subdirectory within log_dir to save tensor data.
            comment (str): Comment to append to the log directory name.
            purge_step (int, optional): Steps to purge when resuming logging.
            max_queue (int): Size of the queue for pending events and summaries.
            flush_secs (int): How often, in seconds, to flush the pending events and summaries.
            filename_suffix (str): Suffix added to all event filenames.
        """
        # Check if the instance has already been initialized to prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.log_dir = log_dir
        self.tensor_data_dir = os.path.join(log_dir, tensor_data_subdir)
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


class BoardFactory(metaclass=SingletonMeta):
    """Factory class for creating board loggers."""

    @staticmethod
    @lru_cache(maxsize=None)
    def get_class(type_lower: str):
        """Retrieve and cache the Board class from the registry."""
        registry_container = RegistryContainer()
        return registry_container.board.get(type_lower)

    def create(self, type: str, **kwargs) -> BoardBase:
        try:
            type_lower = type.lower()
            board_cls = self.get_class(type_lower)
            return board_cls(**kwargs)
        except ValueError as ve:
            logger.error(str(ve))
            raise
        except Exception as e:
            logger.error(f"Failed to create board '{type}': {e}")
            raise
