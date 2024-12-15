import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from simulation.backend.factory import BackendFactory
from simulation.config.config import SimulationConfigManager
from simulation.controller.dispatcher import CommandDispatcher
from simulation.db.factory import DatabaseFactory
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode
from simulation.manager import SimulationManager
from simulation.storage.factory import StorageFactory

console = Console()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="3D Elastic Simulation",
        description="Simulate 3D elastic deformations with contact handling.",
    )
    parser.add_argument(
        "--scenario", type=str, help="Path to the simulation scenario file.", required=True
    )
    return parser.parse_args()


def setup_logging(log_file_path: str = "simulation.log"):
    """Configure pretty logging using rich and save logs to a file.

    Args:
        log_file_path (str): Path to the log file. Defaults to "simulation.log".
    """
    # Set up the Rich handler for console logging
    rich_handler = RichHandler(rich_tracebacks=True, show_path=True)
    rich_handler.setLevel(logging.INFO)

    # Set up the File handler for saving logs to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)  # Save all log levels to the file
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Capture all logs
        handlers=[rich_handler, file_handler],
    )

    logger = logging.getLogger("simulation")
    logger.setLevel(logging.DEBUG)  # Set the level to DEBUG for detailed logs

    logger.info(f"Logging configured. Logs will be saved to {log_file_path}")


# Initialize the logger
logger = logging.getLogger("simulation")


def signal_handler(sig, frame, simulation_manager: SimulationManager):
    """Handle termination signals to gracefully shut down the simulation."""
    logging.info("Interrupt received (signal {}). Shutting down gracefully...".format(sig))
    simulation_manager.stop_simulation()
    sys.exit(0)


def process_commands(simulation_manager: SimulationManager, dispatcher: CommandDispatcher):
    """Process incoming commands and delegate to the SimulationManager."""
    while simulation_manager.simulation_state.get_attribute("running"):
        try:
            # Fetch a command from the backend
            request = simulation_manager.backend.get_command()

            if not request:
                # No command received, continue simulation
                continue

            logger.info(
                SimulationLogMessageCode.COMMAND_RECEIVED.details(
                    f"Processing command: {request.command_name}"
                )
            )

            # Dispatch the command and process response
            response = dispatcher.dispatch(request)
            if response:
                try:
                    simulation_manager.backend.send_response(response)
                    logger.info(
                        SimulationLogMessageCode.COMMAND_EXECUTED.details(
                            f"Command {request.command_name} executed successfully"
                        )
                    )
                except Exception as e:
                    logger.error(
                        SimulationLogMessageCode.COMMAND_EXECUTION_FAILED.details(
                            f"Failed to send response: {e}"
                        )
                    )

        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_PROCESSING_FAILED.details(
                    f"Command processing failed: {e}"
                )
            )
        finally:
            # Small delay to avoid busy-waiting
            time.sleep(0.1)


def main():
    """Main entry point for the simulation."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config_path = Path(args.scenario)
    try:
        config_manager = SimulationConfigManager(config_path=config_path)
        config = config_manager.get()
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(str(e)))
        sys.exit(SimulationErrorCode.CONFIGURATION.value)

    # Initialize Storage
    try:
        storage = StorageFactory.create(config)
        logger.info("Storage initialized successfully.")
    except Exception as e:
        logger.error(SimulationLogMessageCode.STORAGE_FAILED.details(str(e)))
        sys.exit(SimulationErrorCode.STORAGE_INITIALIZATION.value)

    # Initialize Backend
    try:
        backend = BackendFactory.create(config)
        backend.connect()
        logger.info("Backend connected successfully.")
    except Exception as e:
        logger.error(SimulationLogMessageCode.BACKEND_FAILED.details(str(e)))
        sys.exit(SimulationErrorCode.BACKEND_INITIALIZATION.value)

    # Initialize Database
    try:
        database = DatabaseFactory.create(config)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(SimulationLogMessageCode.DATABASE_FAILED.details(str(e)))
        sys.exit(SimulationErrorCode.DATABASE_INITIALIZATION.value)

    # Initialize SimulationManager with external dependencies
    try:
        simulation_manager = SimulationManager(
            scenario=args.scenario, storage=storage, backend=backend, database=database
        )
    except SimulationError as se:
        logger.error(f"SimulationManager initialization failed: {se}")
        sys.exit(se.code)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, simulation_manager))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, simulation_manager))

    # Run the simulation
    try:
        logger.info("Starting simulation...")

        # Run the simulation loop
        simulation_manager.run_simulation()

    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}")
    finally:
        logger.info("Simulation terminated gracefully.")


if __name__ == "__main__":
    main()
