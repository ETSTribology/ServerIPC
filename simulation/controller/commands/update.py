# simulation/commands/send_data.py
import logging

from ..commands.base import BaseCommand, validate_state, SimulationStateMachine
from ..errors import InvalidParametersError, CommandFailedError
from ..model import Request, Response, Status

logger = logging.getLogger(__name__)


class UpdateCommand(BaseCommand):
    """Command to send data (e.g., meshes) to the backend."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            data = request.parameters.get("data")
            key = request.parameters.get("key")

            if not key or data is None:
                raise InvalidParametersError("Key and data must be provided.")

            self.backend.write(key, data)

            message = f"Data sent successfully with key: {key}"
            logger.info(message)
            return Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message=message,
            )
        except InvalidParametersError as e:
            raise e
        except Exception as e:
            logger.error(str(e))
            raise CommandFailedError(str(e))
