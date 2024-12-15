import asyncio
import concurrent.futures
import logging
from typing import Optional

from simulation.controller.commands.factory import CommandFactory
from simulation.controller.errors import CannotConnectError, CommandError
from simulation.controller.model import CommandType, Request, Response, Status

logger = logging.getLogger(__name__)


class CommandDispatcher:
    """Simplified command dispatcher."""

    MAX_RETRIES = 3
    COMMAND_TIMEOUT = 30

    def __init__(self, backend, simulation_manager):
        self.command_factory = CommandFactory(backend, simulation_manager=simulation_manager)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _validate_request(self, request: Request) -> Optional[Response]:
        if not request.command_name:
            logger.error("Command name is required")
            return Response(
                request_id=request.request_id,
                status=Status.INVALID_PARAMETERS.value,
                message="Command name is required",
            )

        try:
            CommandType.get_by_name(request.command_name)
        except ValueError:
            logger.error(f"Invalid command type: {request.command_name}")
            return Response(
                request_id=request.request_id,
                status=Status.INVALID_PARAMETERS.value,
                message=f"Invalid command type: {request.command_name}",
            )

        # Additional parameter checks
        if not isinstance(request.parameters, dict):
            logger.error("Invalid parameters format; expected a dictionary.")
            return Response(
                request_id=request.request_id,
                status=Status.INVALID_PARAMETERS.value,
                message="Parameters must be a dictionary.",
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
            error_message = f"Command execution timed out after {self.COMMAND_TIMEOUT} seconds"
            logger.error(error_message)
            return Response(
                request_id=request.request_id,
                status=Status.ERROR.value,
                message=error_message,
            )

    def dispatch(self, request: Request) -> Response:
        """Synchronous command dispatch with retry capability."""
        validation_error = self._validate_request(request)
        if validation_error:
            return validation_error

        logger.info(f"Dispatching command: {request.command_name}")

        for attempt in range(self.MAX_RETRIES):
            try:
                command = self.command_factory.create(request.command_name, **request.parameters)
                if not command:
                    return self._create_error_response(request, "Unknown command type")

                response = command.execute(request)
                self._log_success(request, attempt)
                return response

            except CommandError as e:
                if isinstance(e, (CannotConnectError)) and attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"Retry attempt {attempt + 1} for command {request.command_name}"
                    )
                    continue
                return self._handle_error(request, e)

            except Exception as e:
                return self._handle_error(request, e)

        # If all retries failed
        return self._create_error_response(request, "Max retries exceeded")

    def _create_error_response(self, request: Request, message: str) -> Response:
        """Create standardized error response."""
        logger.error(message)
        return Response(request_id=request.request_id, status=Status.ERROR.value, message=message)

    def _handle_error(self, request: Request, error: Exception) -> Response:
        """Handle errors and create error responses."""
        error_message = f"Command {request.command_name} failed: {str(error)}"
        logger.error(error_message, exc_info=True)
        return Response(
            request_id=request.request_id, status=Status.ERROR.value, message=error_message
        )

    def _log_success(self, request: Request, attempt: int) -> None:
        """Log successful command execution."""
        logger.info(
            f"Command {request.command_name} executed successfully (attempt {attempt + 1}/{self.MAX_RETRIES})"
        )

    async def _execute_command(self, request: Request) -> Response:
        """Execute command in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.dispatch, request)
