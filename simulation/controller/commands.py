import logging
import sys
from typing import Callable

from simulation.controller.command import Command, CommandRegistry
from simulation.controller.response import ResponseMessage, Status
from simulation.states.state import SimulationState

logger = logging.getLogger(__name__)


@CommandRegistry.register("start", aliases=["run", "begin"])
class StartCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Starts the simulation if it's not already running."""
        if not simulation_state.running:
            simulation_state.running = True
            logger.info("Simulation started.")
            return ResponseMessage(
                request_id=request_id,
                status=Status.SUCCESS,
                message="Simulation started.",
            )
        logger.warning("Simulation is already running.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.WARNING,
            message="Simulation is already running.",
        )


@CommandRegistry.register("pause", aliases=["suspend"])
class PauseCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Pauses the simulation if it's currently running."""
        if simulation_state.running:
            simulation_state.running = False
            logger.info("Simulation paused.")
            return ResponseMessage(
                request_id=request_id,
                status=Status.SUCCESS,
                message="Simulation paused.",
            )
        logger.error("Simulation is not running; cannot pause.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.ERROR,
            message="Simulation is not running; cannot pause.",
        )


@CommandRegistry.register("stop", aliases=["halt", "terminate"])
class StopCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Stops the simulation if it's currently running."""
        if simulation_state.running:
            simulation_state.running = False
            logger.info("Simulation stopped.")
            return ResponseMessage(
                request_id=request_id,
                status=Status.SUCCESS,
                message="Simulation stopped.",
            )
        logger.error("Simulation is not running; cannot stop.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.ERROR,
            message="Simulation is not running; cannot stop.",
        )


@CommandRegistry.register("resume", aliases=["continue"])
class ResumeCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Resumes the simulation if it's paused."""
        if not simulation_state.running:
            simulation_state.running = True
            logger.info("Simulation resumed.")
            return ResponseMessage(
                request_id=request_id,
                status=Status.SUCCESS,
                message="Simulation resumed.",
            )
        logger.warning("Simulation is already running.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.WARNING,
            message="Simulation is already running.",
        )


@CommandRegistry.register("play", aliases=["continue"])
class PlayCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Plays the simulation if it's not already playing."""
        if not simulation_state.running:
            simulation_state.running = True
            logger.info("Simulation playing.")
            return ResponseMessage(
                request_id=request_id,
                status=Status.SUCCESS,
                message="Simulation playing.",
            )
        logger.warning("Simulation is already playing.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.WARNING,
            message="Simulation is already playing.",
        )


@CommandRegistry.register("kill", aliases=["exit", "terminate"])
class KillCommand(Command):
    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Kills the simulation and exits the application."""
        logger.info("Killing simulation.")
        simulation_state.communication_client.close()
        logger.info("Simulation killed.")
        sys.exit()
        # The following return statement is unreachable but required for type consistency
        return ResponseMessage(
            request_id=request_id, status=Status.SUCCESS, message="Simulation killed."
        )


@CommandRegistry.register("reset", aliases=["reinitialize"])
class ResetCommand(Command):
    def __init__(self, reset_function: Callable[[], SimulationState]):
        """Initializes the ResetCommand with a reset function.

        Args:
            reset_function (Callable[[], SimulationState]): A callable that returns a new SimulationState.

        """
        self.reset_function = reset_function

    def execute(self, simulation_state: SimulationState, request_id: str) -> ResponseMessage:
        """Resets the simulation state.

        Args:
            simulation_state (SimulationState): The current simulation state.
            request_id (str): The unique identifier for the request.

        Returns:
            ResponseMessage: The response after resetting the simulation.

        """
        logger.info("Resetting simulation.")
        new_state = self.reset_function()
        simulation_state.update_from(new_state)
        simulation_state.running = False
        logger.info("Simulation reset complete.")
        return ResponseMessage(
            request_id=request_id,
            status=Status.SUCCESS,
            message="Simulation reset complete.",
        )
