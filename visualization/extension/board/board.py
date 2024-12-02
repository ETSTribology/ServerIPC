import abc
from typing import Any, Optional


class Board(abc.ABC):
    """Abstract base class for board loggers."""

    @abc.abstractmethod
    def log_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None) -> None:
        """Logs a scalar value."""
        pass

    @abc.abstractmethod
    def log_text(self, tag: str, text_string: str, global_step: Optional[int] = None) -> None:
        """Logs a text string."""
        pass

    @abc.abstractmethod
    def log_image(
        self, tag: str, img_tensor: Any, global_step: Optional[int] = None, dataformats: str = "CHW"
    ) -> None:
        """Logs an image."""
        pass

    @abc.abstractmethod
    def log_figure(
        self,
        tag: str,
        figure: Any,
        global_step: Optional[int] = None,
        close: bool = True,
        walltime: Optional[float] = None,
    ) -> None:
        """Logs a matplotlib figure."""
        pass

    @abc.abstractmethod
    def log_graph(self, model: Any, input_to_model: Any) -> None:
        """Logs a model graph."""
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        """Flushes any pending logs."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger and releases resources."""
        pass
