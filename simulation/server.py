import argparse
import logging
import signal
import sys

from rich.console import Console
from rich.logging import RichHandler

from simulation.manager import SimulationManager

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


def main():
    """Main entry point for the simulation."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_arguments()

    # Initialize SimulationManager
    simulation_manager = SimulationManager(args.scenario)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, simulation_manager))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, simulation_manager))

    # Run the simulation
    try:
        logger.info("Starting simulation...")
        simulation_manager.run_simulation()
    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}")
    finally:
        logger.info("Simulation terminated gracefully.")


if __name__ == "__main__":
    main()
