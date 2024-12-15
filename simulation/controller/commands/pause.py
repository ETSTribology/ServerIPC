import logging


from simulation.controller.commands.base import BaseCommand, ICommand, SimulationStateMachine, validate_state
from simulation.controller.model import Request, Response, Status,  CommandType
from simulation.controller.errors import CommandFailedError


logger = logging.getLogger(__name__)


class PauseCommand(BaseCommand):
    @validate_state([SimulationStateMachine.RUNNING])
    def _execute_impl(self, request: Request) -> Response:
        try:
            self._update_state(SimulationStateMachine.PAUSED)
            message = CommandType.PAUSE.description
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