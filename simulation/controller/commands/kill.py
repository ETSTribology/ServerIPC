# simulation/commands/kill.py
import logging


from simulation.controller.commands.base import BaseCommand, ICommand, SimulationStateMachine, validate_state
from simulation.controller.errors import CommandFailedError
from simulation.controller.model import Request, Response, Status

logger = logging.getLogger(__name__)


class KillCommand(BaseCommand):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend

    def _execute_impl(self, request: Request) -> Response:
        try:
            message = "Simulation is shutting down gracefully."
            logger.info(message)
            self.backend.initiate_shutdown()
            self._update_state(SimulationStateMachine.STOPPED)
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(str(e))
            raise CommandFailedError(str(e))

    def _update_state(self, new_state: SimulationStateMachine):
        """Update simulation state."""
        self.previous_state = self.current_state
        self.current_state = new_state
