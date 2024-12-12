import argparse
import logging
import signal
import sys

from simulation.manager import SimulationManager


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


def setup_logging():
    """Configure logging for the simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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

    # Initialize the simulation with the provided scenario
    try:
        logger.info(f"Initializing simulation with scenario: {args.scenario}")
        simulation_manager.initialize_simulation(scenario=args.scenario)
    except Exception as e:
        logger.error(f"Failed to initialize the simulation: {e}")
        sys.exit(1)

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
