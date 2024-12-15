# simulation/commands/reset.py
import logging
from typing import Callable

from simulation.controller.errors import CommandFailedError
from simulation.controller.commands.base import BaseCommand, SimulationStateMachine
from simulation.controller.model import Request, Response, Status

logger = logging.getLogger(__name__)


class ResetCommand(BaseCommand):
    def __init__(self, reset_function: Callable[[], None]):
        super().__init__()
        self.reset_function = reset_function

    def _execute_impl(self, request: Request) -> Response:
        try:
            self.reset_function()
            self._update_state(SimulationStateMachine.STOPPED)
            message = "Simulation reset complete."
            logger.info(message)
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except Exception as e:
            logger.error(str(e))
            raise CommandFailedError(f"Reset failed: {str(e)}")

    def _update_state(self, new_state: SimulationStateMachine):
        """Update simulation state."""
        self.previous_state = self.current_state
        self.current_state = new_state
