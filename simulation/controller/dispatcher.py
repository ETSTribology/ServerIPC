import asyncio
import concurrent.futures
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from time import perf_counter
from typing import Dict, Optional

from simulation.controller.commands import CannotConnectError, CommandError
from simulation.controller.factory import CommandFactory
from simulation.controller.history import CommandHistory
from simulation.controller.model import CommandType, Request, Response, Status
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)


@dataclass
class CommandMetrics:
    execution_time: float
    attempts: int
    status: str
    timestamp: str


class CommandDispatcher:
    """Enhanced command dispatcher with retry, timeout, and metrics capabilities."""

    MAX_RETRIES = 3
    COMMAND_TIMEOUT = 30  # seconds

    def __init__(self, history: CommandHistory):
        self.command_factory = CommandFactory(history)
        self.history = history
        self._metrics: Dict[str, CommandMetrics] = {}
        self._metrics_lock = Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    @contextmanager
    def _track_execution_time(self, command_type: str):
        """Context manager for tracking command execution time."""
        start_time = perf_counter()
        try:
            yield
        finally:
            execution_time = perf_counter() - start_time
            self._update_metrics(command_type, execution_time)

    def _update_metrics(self, command_type: str, execution_time: float, status: str = "SUCCESS"):
        """Update command execution metrics."""
        with self._metrics_lock:
            if command_type not in self._metrics:
                self._metrics[command_type] = CommandMetrics(
                    execution_time=execution_time,
                    attempts=1,
                    status=status,
                    timestamp=datetime.now().isoformat(),
                )
            else:
                metric = self._metrics[command_type]
                metric.execution_time = (metric.execution_time + execution_time) / 2
                metric.attempts += 1
                metric.status = status
                metric.timestamp = datetime.now().isoformat()

    def _validate_request(self, request: Request) -> Optional[Response]:
        """Validate incoming request parameters."""
        if not request.command_name:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details("Command name is required"))
            return Response(
                request_id=request.request_id,
                status=Status.INVALID_PARAMETERS.value,
                message="Command name is required",
            )

        try:
            CommandType.get_by_name(request.command_name)
        except ValueError:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Invalid command type: {request.command_name}"))
            return Response(
                request_id=request.request_id,
                status=Status.INVALID_PARAMETERS.value,
                message=f"Invalid command type: {request.command_name}",
            )

        return None

    async def dispatch_async(self, request: Request) -> Response:
        """Asynchronous command dispatch with timeout."""
        validation_error = self._validate_request(request)
        if validation_error:
            return validation_error

        try:
            return await asyncio.wait_for(
                self._execute_command(request), timeout=self.COMMAND_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Command execution timed out after {self.COMMAND_TIMEOUT} seconds"))
            return Response(
                request_id=request.request_id,
                status=Status.ERROR.value,
                message=f"Command execution timed out after {self.COMMAND_TIMEOUT} seconds",
            )

    def dispatch(self, request: Request) -> Response:
        """Synchronous command dispatch with retry capability."""
        validation_error = self._validate_request(request)
        if validation_error:
            return validation_error

        logger.info(SimulationLogMessageCode.COMMAND_STARTED.details(f"Dispatching command: {request.command_name}"))

        for attempt in range(self.MAX_RETRIES):
            with self._track_execution_time(request.command_name):
                try:
                    command = self.command_factory.create(
                        request.command_name, **request.parameters
                    )
                    if not command:
                        return self._create_error_response(request, "Unknown command type")

                    response = command.execute(request)
                    self._log_success(request, attempt)
                    return response

                except CannotConnectError as e:
                    if attempt < self.MAX_RETRIES - 1:
                        logger.warning(SimulationLogMessageCode.COMMAND_RETRY.details(f"Retry attempt {attempt + 1} for command {request.command_name}"))
                        continue
                    return self._handle_error(request, e, Status.CANNOT_CONNECT.value)

                except InvalidParametersError as e:
                    return self._handle_error(request, e, Status.INVALID_PARAMETERS.value)

                except CommandError as e:
                    return self._handle_error(request, e, Status.COMMAND_FAILED.value)

                except Exception as e:
                    return self._handle_error(request, e, Status.ERROR.value)

    def _create_error_response(self, request: Request, message: str) -> Response:
        """Create standardized error response."""
        logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(message))
        return Response(request_id=request.request_id, status=Status.ERROR.value, message=message)

    def _handle_error(self, request: Request, error: Exception, status: str) -> Response:
        """Handle command errors with logging."""
        logger.error(SimulationLogMessageCode.COMMAND_FAILED.details(f"Command {request.command_name} failed: {str(error)}"))
        self._update_metrics(request.command_name, 0, status)
        return Response(request_id=request.request_id, status=status, message=str(error))

    def _log_success(self, request: Request, attempt: int) -> None:
        """Log successful command execution."""
        logger.info(SimulationLogMessageCode.COMMAND_SUCCESS.details(
            f"Command {request.command_name} executed successfully (attempt {attempt + 1}/{self.MAX_RETRIES})"
        ))

    async def _execute_command(self, request: Request) -> Response:
        """Execute command in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.dispatch, request)

    def get_metrics(self) -> Dict[str, CommandMetrics]:
        """Get command execution metrics."""
        with self._metrics_lock:
            return dict(self._metrics)