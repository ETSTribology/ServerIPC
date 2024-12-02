import os
import torch
from typing import Optional, Any, Dict
from visualization.extension.board.board import Board
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

from visualization.storage.factory import StorageFactory

logger = logging.getLogger(__name__)


class TensorBoard(Board):
    """A TensorBoard logger for scalar, text, graph, images, histograms, and Plotly logging."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TensorBoard logger.

        Args:
            log_dir (str): Directory for TensorBoard logs.
        """
        logger.info("Initializing TensorBoard logger...")
        self.log_dir = StorageFactory.create(config).get_directory("board")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(f"TensorBoard initialized with log directory: {self.log_dir}")

    def log_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None) -> None:
        """Logs a scalar value."""
        self.writer.add_scalar(tag, scalar_value, global_step)
        logger.debug(f"Logged scalar: tag={tag}, value={scalar_value}, step={global_step}")

    def log_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        """Logs a text string."""
        self.writer.add_text(tag, text_string, global_step)
        logger.debug(f"Logged text: tag={tag}, text={text_string}, step={global_step}")

    def log_histogram(self, tag: str, values: Any, global_step: Optional[int] = None, bins: Optional[str] = "tensorflow") -> None:
        """Logs a histogram of values."""
        self.writer.add_histogram(tag, values, global_step, bins=bins)
        logger.debug(f"Logged histogram: tag={tag}, step={global_step}")

    def log_image(self, tag: str, img_tensor: torch.Tensor, global_step: Optional[int] = None, dataformats: str = "CHW") -> None:
        """
        Logs an image.

        Args:
            tag (str): Data identifier.
            img_tensor (torch.Tensor): Image data tensor.
            global_step (Optional[int]): Global step value to record.
            dataformats (str): Image data format.
        """
        self.writer.add_image(tag, img_tensor, global_step, dataformats)
        logger.debug(f"Logged image: tag={tag}, step={global_step}")

    def log_figure(self, tag: str, figure: Any, global_step: Optional[int] = None, close: bool = True, walltime: Optional[float] = None) -> None:
        """Logs a matplotlib figure."""
        self.writer.add_figure(tag, figure, global_step, close=close, walltime=walltime)
        logger.debug(f"Logged figure: tag={tag}, step={global_step}")

    def log_graph(self, model: torch.nn.Module, input_to_model: torch.Tensor) -> None:
        """
        Logs a model graph.

        Args:
            model (torch.nn.Module): The model to be visualized.
            input_to_model (torch.Tensor): Example input tensor to generate the graph.
        """
        try:
            self.writer.add_graph(model, input_to_model)
            logger.info("Logged model graph.")
        except Exception as e:
            logger.error(f"Failed to log model graph: {e}")

    def flush(self) -> None:
        """Flushes any pending logs."""
        self.writer.flush()
        logger.debug("Flushed pending logs.")

    def close(self) -> None:
        """Closes the logger and releases resources."""
        self.writer.close()
        logger.info("Closed TensorBoard logger.")
