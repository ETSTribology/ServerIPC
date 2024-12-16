import threading
import logging
import time
import numpy as np
from typing import Optional, Dict, Any
from simulation.controller.dispatcher import CommandDispatcher
from simulation.controller.model import Response, Status, Request
from simulation.db.db import DatabaseBase
from simulation.db.factory import DatabaseFactory
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles commands and manages data updates for the simulation."""

    def __init__(self, backend, simulation_manager):
        self.backend = backend
        self.simulation_manager = simulation_manager
        self.dispatcher = CommandDispatcher(backend=backend, simulation_manager=simulation_manager)
        self._stop_event = threading.Event()
        self._command_thread = threading.Thread(target=self._command_loop, daemon=True)

        # Validate and set config name
        self.config = simulation_manager.config
        self.config_name = self.config.get("name", "simulation")

        # Initialize database IDs
        self.config_id = None
        self.run_id = None
        self._initialize_config_and_run()

    def _initialize_config_and_run(self):
        """
        Initializes config_id and creates a new run_id by interacting with the database.
        """
        current_db: Optional[DatabaseBase] = DatabaseFactory.get_current_db()
        if not current_db:
            logger.error("No database instance available to initialize config and run.")
            return

        try:
            # Fetch or create Config
            existing_config = current_db.client.config.find_first(where={"name": self.config_name})
            if not existing_config:
                # Create a new Config entry
                new_config = current_db.client.config.create(
                    data={
                        "name": self.config_name
                    }
                )
                self.config_id = new_config.id
                logger.info(f"New Config created with ID: {self.config_id}")
            else:
                self.config_id = existing_config.id
                logger.info(f"Using existing Config with ID: {self.config_id}")

            # Always create a new Run for the existing Config
            new_run = current_db.client.run.create(
                data={
                    "configId": self.config_id
                }
            )
            self.run_id = new_run.id
            logger.info(f"New Run created with ID: {self.run_id}")

        except Exception as e:
            logger.error(f"Failed to initialize Config and Run: {e}")

    def process_command(self, request: Request):
        """
        Process a command using the backend command handler.

        Args:
            request (Request): The command request object.
        """
        self.backend_command_handler(request)

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

    def send_data_update(self, backend_data: Dict[str, Any], db_data: Dict[str, Any]):
        """
        Sends simulation data updates to the backend and stores it in the database.

        Args:
            backend_data (Dict[str, Any]): Data to be sent to the backend.
            db_data (Dict[str, Any]): Data to be stored in the database.
        """
        try:
            # Send data to the backend
            response = Response(
                request_id=f"{db_data['timeStep']}-update",
                status=Status.SUCCESS.value,
                message="Data update sent",
                data=backend_data,
            )
            self.backend.send_response(response)

            # Save data to the database in a separate thread
            threading.Thread(target=self.database_command_handler, args=(db_data,), daemon=True).start()

            logger.info(f"Data update sent successfully for time step {db_data['timeStep']}.")
        except Exception as e:
            logger.error(f"Failed to send data update: {e}")
            
    def database_command_handler(self, state_data: Dict[str, Any]):
        """
        Saves simulation state data to the database using Prisma.

        Args:
            state_data (Dict[str, Any]): The state data to save in the database.
        """
        current_db: Optional[DatabaseBase] = DatabaseFactory.get_current_db()
        if not current_db:
            logger.error("No database instance available to save the data.")
            return

        try:

            # Separate velocity, acceleration, and position into x, y, z components
            velocity = np.array(state_data.pop("velocity", []))
            acceleration = np.array(state_data.pop("acceleration", []))
            position = np.array(state_data.pop("positions", []))  # Fixed key

            # Create State entry first with timeStep
            state_entry = current_db.client.state.create(
                data={
                    "runId": self.run_id,  # Ensure runId is correctly passed
                    "timeStep": state_data.get("timeStep", 0),
                    "timeElapsed": state_data.get("timeElapsed", 0),
                    "num_elements": state_data.get("num_elements", 0),
                }
            )
            state_id = state_entry.id  # Get the id of the created state entry

            # Create Position, Velocity, and Acceleration entries for each node
            elements_data = []
            for i in range(position.shape[0]):
                # Create Position entry
                position_entry = current_db.client.position.create(
                    data={
                        "x": position[i, 0],
                        "y": position[i, 1],
                        "z": position[i, 2],
                    }
                )

                # Create Velocity entry
                velocity_entry = current_db.client.velocity.create(
                    data={
                        "x": velocity[i, 0],
                        "y": velocity[i, 1],
                        "z": velocity[i, 2],
                    }
                )

                # Create Acceleration entry
                acceleration_entry = current_db.client.acceleration.create(
                    data={
                        "x": acceleration[i, 0],
                        "y": acceleration[i, 1],
                        "z": acceleration[i, 2],
                    }
                )

                # Create Element entry and associate it with the created State and Position/Velocity/Acceleration
                element_entry = current_db.client.element.create(
                    data={
                        "stateId": state_id,  # Link to the created state entry
                        "positionId": position_entry.id,
                        "velocityId": velocity_entry.id,
                        "accelerationId": acceleration_entry.id,
                    }
                )
                elements_data.append(element_entry)  # Store the created element

            logger.info(f"Created {len(elements_data)} elements for state {state_id}.")
            logger.info(f"State data for step {state_data['timeStep']} saved successfully.")
            
        except Exception as e:
            logger.error(f"Unexpected error during database operation: {e}")
            logger.debug("Ensure all fields match the Prisma schema and check database migrations.")

    def backend_command_handler(self, request: Request):
        """
        Handles backend commands.

        Args:
            request (Request): The command request object containing command details.
        """
        logger.info(f"Processing backend command: {request.command_name}")
        try:
            if request.command_name == "start":
                self.simulation_manager.resume_simulation()
            elif request.command_name == "pause":
                self.simulation_manager.pause_simulation()
            else:
                logger.warning(f"Unknown command: {request.command_name}")

            # Send response back
            response = Response(
                request_id=request.request_id,
                status=Status.SUCCESS.value,
                message=f"Command '{request.command_name}' executed successfully",
            )
            self.backend.send_response(response)
        except Exception as e:
            logger.error(f"Failed to process command '{request.command_name}': {e}")
            response = Response(
                request_id=request.request_id,
                status=Status.FAILURE.value,
                message=f"Command '{request.command_name}' failed: {str(e)}",
            )
            self.backend.send_response(response)

    def _command_loop(self):
        """Thread loop for handling commands."""
        logger.info("Command handling thread started.")
        while not self._stop_event.is_set():
            try:
                is_running = self.simulation_manager.simulation_state.get_attribute("running")
                if not is_running:
                    logger.info("Simulation is paused. Waiting for commands...")
                    time.sleep(1)
                    continue

                request = self.backend.get_command()
                if request:
                    self.process_command(request)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in command handling loop: {e}")

        logger.info("Command handling thread stopped.")
