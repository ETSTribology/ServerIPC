import logging
import signal
import sys

from loop import SimulationManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def signal_handler(sig, frame, simulation_manager: SimulationManager):
    """Handles system signals (e.g., SIGINT, SIGTERM) to gracefully shut down the simulation.

    Parameters
    ----------
    sig : int
        Signal number received.
    frame : FrameType
        Current stack frame.
    simulation_manager : SimulationManager
        The simulation manager handling the simulation.

    """
    logger.info("Interrupt received, shutting down...")
    try:
        communication_client = simulation_manager.simulation_state.get_attribute(
            "communication_client"
        )
        if communication_client:
            communication_client.close()
            logger.info("Communication client closed successfully.")
    except Exception as e:
        logger.error(f"Error while closing communication client: {e}")
    sys.exit(0)


def main():
    """Main entry point for the simulation server.
    Initializes the simulation manager, sets up signal handlers, and starts the simulation loop.
    """
    simulation_manager = SimulationManager()

    # Initialize the simulation
    try:
        simulation_manager.initialize_simulation()
    except Exception as e:
        logger.error(f"Failed to initialize the simulation: {e}")
        sys.exit(1)

    # Retrieve the communication client from the simulation state
    communication_client = simulation_manager.simulation_state.get_attribute(
        "communication_client"
    )
    if communication_client is None:
        logger.error("SimulationState does not contain a communication client.")
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    signal.signal(
        signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, simulation_manager)
    )
    signal.signal(
        signal.SIGTERM,
        lambda sig, frame: signal_handler(sig, frame, simulation_manager),
    )

    try:
        simulation_manager.run_simulation()
    except Exception as e:
        logger.error(f"An error occurred during simulation: {e}")
    finally:
        # Ensure communication client is closed
        try:
            if communication_client:
                communication_client.close()
                logger.info("Communication client closed successfully.")
        except Exception as e:
            logger.error(f"Error while closing communication client: {e}")
        logger.info("Simulation terminated gracefully.")


if __name__ == "__main__":
    main()
