import logging

from simulation.controller.commands.base import BaseCommand, SimulationStateMachine, validate_state
from simulation.controller.errors import CommandFailedError
from simulation.controller.model import Request, Response, Status, CommandType



logger = logging.getLogger(__name__)


class StopCommand(BaseCommand):
    @validate_state([SimulationStateMachine.RUNNING, SimulationStateMachine.PAUSED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            self._update_state(SimulationStateMachine.STOPPED)
            message = CommandType.STOP.description
            logger.info(message)
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
