import threading
import logging
import numpy as np
import time
from simulation.controller.dispatcher import CommandDispatcher
from simulation.controller.model import Response, Request, Status

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles commands and manages data updates for the simulation."""

    def __init__(self, backend, simulation_manager):
        self.backend = backend
        self.simulation_manager = simulation_manager
        self.dispatcher = CommandDispatcher(backend=backend, simulation_manager=simulation_manager)
        self._stop_event = threading.Event()
        self._command_thread = threading.Thread(target=self._command_loop, daemon=True)

    def process_command(self, request):
        logger.info(f"Processing command: {request.command_name}")
        try:
            # Mock command processing
            logger.info(f"Command {request.command_name} executed successfully")
            response = Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message="Command executed",
            )
            self.backend.send_response(response)
        except Exception as e:
            logger.error(f"Failed to process command: {e}")

    def _command_loop(self):
        """Thread loop for handling commands."""
        logger.info("Command handling thread started.")
        while not self._stop_event.is_set():
            try:
                if not self.simulation_manager.simulation_state.get_attribute("running"):
                    logger.info("Simulation is paused. Waiting for commands...")
                    time.sleep(1)
                    continue

                request = self.backend.get_command()
                if request:
                    self.process_command(request)
                else:
                    logger.info("No command received. Waiting...")

                time.sleep(0.1)  # Avoid busy-waiting
            except Exception as e:
                logger.error(f"Error in command handling loop: {e}")

        logger.info("Command handling thread stopped.")

    def start_command_listener(self):
        """Start the command listener thread."""
        if not self._command_thread.is_alive():
            self._stop_event.clear()
            self._command_thread = threading.Thread(target=self._command_loop, daemon=True)
            self._command_thread.start()
            logger.info("Command listener thread started.")

    def stop_command_listener(self):
        """Stop the command listener thread."""
        self._stop_event.set()
        if self._command_thread.is_alive():
            self._command_thread.join(timeout=2)
            logger.info("Command listener thread stopped.")

    def send_data_update(self, x: np.ndarray, BX: np.ndarray, step: int):
        try:
            mesh_data = {
                "timestamp": time.time(),
                "step": step,
                "x": x.tolist(),
                "BX": BX.tolist(),
            }
            response = Response(
                request_id=f"{step}-update",
                status=Status.SUCCESS.value,
                message="Data update sent",
                data=mesh_data,
            )
            self.backend.send_response(response)
            logger.info(f"Data update sent successfully at step {step}")
        except Exception as e:
            logger.error(f"Failed to send data update: {e}")