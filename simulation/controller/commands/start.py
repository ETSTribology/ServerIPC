import logging
from typing import Optional
from simulation.controller.errors import CannotConnectError, CommandFailedError
from simulation.controller.commands.base import BaseCommand, ICommand, SimulationStateMachine, validate_state
from simulation.controller.model import Request, Response, Status
from simulation.controller.model import CommandType

logger = logging.getLogger(__name__)


class StartCommand(BaseCommand):
    @validate_state([SimulationStateMachine.STOPPED])
    def _execute_impl(self, request: Request) -> Response:
        try:
            success = self._connect()
            if not success:
                error_message = "Failed to connect to simulation engine."
                logger.error(error_message)
                raise CannotConnectError(error_message)
            self._update_state(SimulationStateMachine.RUNNING)
            message = CommandType.START.description
            logger.info(message)
            return Response(
                request_id=request.request_id, status=Status.SUCCESS.value, message=message
            )
        except CannotConnectError as e:
            raise e
        except Exception as e:
            logger.error(str(e))
            raise CommandFailedError(str(e))

    def _connect(self) -> bool:
        self.simulation_manager.start_simulation()
        return {"status": "success", "message": "Simulation started."}

    def _update_state(self, new_state: SimulationStateMachine):
        """Update simulation state."""
        self.previous_state = self.current_state