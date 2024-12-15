import logging

from simulation.controller.commands.base import BaseCommand, ICommand, SimulationStateMachine, validate_state
from simulation.controller.errors import CommandFailedError
from simulation.controller.model import Request, Response, Status

logger = logging.getLogger(__name__)

class StatusCommand(BaseCommand):
    """Command to retrieve the status of the backend."""

    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            backend_status = self.backend.get_status()

            message = "Backend status retrieved successfully."
            logger.info(message)
            return Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message=message,
                data=backend_status,
            )
        except Exception as e:
            logger.error(str(e))
            raise CommandFailedError(str(e))
