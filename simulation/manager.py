import asyncio
import logging
import sys
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np

from simulation.config.config import SimulationConfigManager
from simulation.controller.handler import CommandHandler
from simulation.core.contact.barrier import BarrierFactory
from simulation.core.contact.ccd import CCDFactory
from simulation.core.math.gradient import GradientFactory
from simulation.core.math.hessian import HessianFactory
from simulation.core.math.potential import PotentialFactory
from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.solvers.line_search import LineSearchFactory
from simulation.core.solvers.linear import LinearSolverBase, LinearSolverFactory
from simulation.core.solvers.optimizer import OptimizerBase, OptimizerFactory
from simulation.initializer import SimulationInitializer
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode
from simulation.states.state import SimulationState

logger = logging.getLogger(__name__)


class SimulationManager:
    def __init__(self, scenario: str, storage: Any, backend: Any, database: Any):
        self.scenario = scenario
        self.config_path = Path(scenario)
        self.config = self.load_configuration()
        self.storage = storage
        self.backend = backend
        self.database = database
        self.initialize_factories()

        self.simulation_state = self.initialize_simulation()
        self.line_searcher = self.setup_line_searcher()
        self.linear_solver = self.setup_linear_solver()
        self.optimizer = self.setup_optimizer()
        self.params = self.setup_parameters()
        self.potentials = self.setup_potentials()
        self.gradient = self.setup_gradient()
        self.hessian = self.setup_hessian()
        self.barrier_updater = self.setup_barrier_updater()
        self.ccd = self.setup_ccd()

        self.command_handler = CommandHandler(backend=self.backend, simulation_manager=self)

        self.backend.set_command_handler(self.command_handler.process_command)

        self.total_time = self.config.get("time", {}).get("total", 1.0)
        self.time_step = self.config.get("time", {}).get("step", 0.01)
        self.max_steps = int(self.total_time / self.time_step)

        self.current_step = 0
        self._shutdown_initiated = False

    def load_configuration(self) -> Dict[str, Any]:
        logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED)
        try:
            config_manager = SimulationConfigManager(config_path=self.config_path)
            config = config_manager.get()
            logger.debug(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"{config}"))
            return config
        except Exception as e:
            logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.CONFIGURATION, "Failed to load configuration", str(e)
            )

    def initialize_factories(self):
        logger.info(SimulationLogMessageCode.FACTORIES_INITIALIZED)
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()
        self.optimizer_factory = OptimizerFactory()

    def initialize_simulation(self) -> SimulationState:
        logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
        try:
            simulation_state = SimulationInitializer(scenario=self.scenario).initialize_simulation()
            logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
            return simulation_state
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_INITIALIZATION_ABORTED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.SIMULATION_LOOP, "Failed to initialize simulation", str(e)
            )

    def setup_parameters(self) -> ParametersBase:
        logger.info("Setting up simulation parameters.")
        required_attrs = [
            "mesh",
            "x",
            "v",
            "acceleration",
            "mass_matrix",
            "hep",
            "dt",
            "cmesh",
            "cconstraints",
            "fconstraints",
            "materials",
            "element_materials",
            "dhat",
            "dmin",
            "mu",
            "epsv",
            "barrier_potential",
            "friction_potential",
        ]
        attributes = {}
        for attr in required_attrs:
            try:
                attributes[attr] = self.simulation_state.get_attribute(attr, default=None)
                if attributes[attr] is None:
                    logger.warning(f"Attribute '{attr}' is missing or None.")
            except Exception as e:
                logger.error(f"Error retrieving attribute '{attr}': {e}")
                raise SimulationError(
                    SimulationErrorCode.PARAMETERS_SETUP,
                    f"Failed to retrieve attribute '{attr}'",
                    str(e),
                )

        try:
            parameters = Parameters(
                mesh=attributes["mesh"],
                config=self.config,
                dt=self.config.get("time", {}).get("step", 0.01),
                xt=attributes.get("x", np.zeros(0)),
                vt=attributes.get("v", np.zeros(0)),
                a=attributes.get("acceleration", np.zeros(0)),
                M=attributes.get("mass_matrix", np.zeros(0)),
                hep=attributes["hep"],
                cmesh=attributes["cmesh"],
                cconstraints=attributes["cconstraints"],
                fconstraints=attributes["fconstraints"],
                materials=attributes["materials"],
                element_materials=attributes["element_materials"],
                barrier_potential=attributes["barrier_potential"],
                friction_potential=attributes["friction_potential"],
                broad_phase_method=self.config.get("contact", {}).get("broad_phase", "brute_force"),
            )
            logger.info("Simulation parameters set up successfully.")
            return parameters
        except Exception as e:
            logger.error(f"Failed to initialize parameters: {e}")
            raise SimulationError(
                SimulationErrorCode.PARAMETERS_SETUP, "Failed to initialize parameters", str(e)
            )

    def setup_potentials(self) -> Any:
        logger.info("Setting up potentials.")
        try:
            potentials = PotentialFactory.create(params=self.params)
            logger.info("Potentials set up successfully.")
            return potentials
        except Exception as e:
            logger.error(f"Failed to setup potentials: {e}")
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to setup potentials", str(e)
            )

    def setup_gradient(self, *args, **kwargs) -> Any:
        """
        Setup gradient with parameter validation.

        Accepts additional positional and keyword arguments to prevent TypeError
        when called with extra parameters.
        """
        logger.info("Setting up gradient.")
        try:
            # If a params object is passed, use it; otherwise, use self.params
            if args:
                params = args[0]
                logger.debug("Received params argument for gradient setup.")
            else:
                params = self.params

            gradient = GradientFactory.create(params=params)
            # Test gradient with dummy input
            test_x = np.zeros(len(params.xt))
            test_grad = gradient(test_x)
            if not isinstance(test_grad, np.ndarray):
                raise ValueError("Gradient must return numpy array")
            self.simulation_state.update_attribute("gradient", gradient)
            logger.info("Gradient setup successful.")
            return gradient
        except Exception as e:
            logger.error(f"Failed to setup gradient: {e}")
            raise SimulationError(
                SimulationErrorCode.GRADIENT_SETUP, "Failed to setup gradient", str(e)
            )

    def setup_hessian(self, *args, **kwargs) -> Any:
        logger.info("Setting up hessian.")
        try:
            hessian = HessianFactory.create(params=self.params)
            self.simulation_state.update_attribute("hessian", hessian)
            logger.info("Hessian set up successfully.")
            return hessian
        except Exception as e:
            logger.error(f"Failed to setup hessian: {e}")
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP, "Failed to setup hessian", str(e)
            )

    def setup_barrier_updater(self, *args, **kwargs) -> Any:
        logger.info("Setting up barrier updater.")
        try:
            barrier_updater = BarrierFactory.create_updater(params=self.params)
            self.simulation_state.update_attribute("barrier_updater", barrier_updater)
            logger.info("Barrier updater set up successfully.")
            return barrier_updater
        except Exception as e:
            logger.error(f"Failed to setup barrier updater: {e}")
            raise SimulationError(
                SimulationErrorCode.BARRIER_UPDATER_SETUP, "Failed to setup barrier updater", str(e)
            )

    def setup_ccd(self, *args, **kwargs) -> Any:
        logger.info("Setting up CCD solver.")
        try:
            ccd = CCDFactory.create(params=self.params)
            self.simulation_state.update_attribute("ccd", ccd)
            logger.info("CCD solver set up successfully.")
            return ccd
        except Exception as e:
            logger.error(f"Failed to setup CCD solver: {e}")
            raise SimulationError(
                SimulationErrorCode.CCD_SETUP, "Failed to setup CCD solver", str(e)
            )

    def setup_line_searcher(self) -> Any:
        try:
            solver_config = self.config.get("solver", {}).get("optimizer", {})
            method = solver_config.get("line_search", "armijo").lower()

            valid_methods = {"armijo", "wolfe", "strong_wolfe", "parallel", "backtracking"}
            if method not in valid_methods:
                logger.warning(f"Invalid line search method '{method}', falling back to 'armijo'")
                method = "armijo"

            grad_required = method in {"wolfe", "strong_wolfe"}

            hep = self.simulation_state.get_attribute("hep")
            if hep is None:
                raise ValueError("Missing required objective function 'hep'")

            gradient = None
            if grad_required:
                gradient = self.simulation_state.get_attribute("gradient")
                if gradient is None:
                    raise ValueError(f"Gradient function required for {method} line search")

            line_searcher = self.line_search_factory.create(
                config=self.config, f=hep, grad_f=gradient
            )

            logger.info(
                f"Line searcher '{method}' configured with gradient required: {grad_required}"
            )
            return line_searcher

        except Exception as e:
            logger.error(f"Failed to setup line searcher: {str(e)}")
            raise SimulationError(
                SimulationErrorCode.LINE_SEARCH_SETUP, "Failed to setup line searcher", str(e)
            )

    def setup_linear_solver(self) -> LinearSolverBase:
        try:
            method = self.config.get("linear", {}).get("solver", "direct")
            method = "direct" if method == "default" else method
            dofs = self.simulation_state.get_attribute("degrees_of_freedom")
            linear_solver = self.linear_solver_factory.create(config=self.config, dofs=dofs)
            logger.info(f"Linear solver set up successfully with method: {method}")
            return linear_solver
        except Exception as e:
            logger.error(SimulationLogMessageCode.LINEAR_SOLVER_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.LINEAR_SOLVER_SETUP, "Failed to setup linear solver", str(e)
            )

    def setup_optimizer(self) -> OptimizerBase:
        """Setup optimizer with proper gradient handling."""
        try:
            # Get optimizer configuration
            optimizer_config = self.config.get("solver", {}).get("optimization", {})
            optimizer_type = optimizer_config.get("solver", "newton").lower()

            # Get required components with validation
            hep = self.simulation_state.get_attribute("hep")
            if hep is None:
                raise ValueError("Objective function 'hep' is missing")

            # Get parameters
            params = self.simulation_state.get_attribute("params")
            if params is None:
                params = self.setup_parameters()
                self.simulation_state.update_attribute("params", params)

            # Setup gradient with parameters
            gradient = self.setup_gradient(params)
            self.simulation_state.update_attribute("gradient", gradient)

            # Continue with optimizer creation
            linear_solver_config = self.config.get("solver", {}).get("linear", {})
            dofs = self.simulation_state.get_attribute("degrees_of_freedom")

            optimizer = self.optimizer_factory.create(
                config=self.config.get("solver", {}),
                f=hep,
                grad_f=gradient,
                linear_solver_config=linear_solver_config,
                dofs=dofs,
            )

            logger.info(f"Optimizer setup complete with type: {optimizer_type}")
            return optimizer

        except Exception as e:
            logger.error(f"Failed to setup optimizer: {str(e)}")
            raise SimulationError(
                SimulationErrorCode.OPTIMIZER_SETUP, "Failed to setup optimizer", str(e)
            )

    def update_simulation_state(self, params, potential, gradient, hessian, barrier_updater, ccd):
        try:
            self.simulation_state.update_attributes(
                {
                    "params": params,
                    "potential": potential,
                    "gradient": gradient,
                    "hessian": hessian,
                    "barrier_updater": barrier_updater,
                    "ccd": ccd,
                }
            )
            logger.info(
                "Simulation state updated with parameters, potential, gradient, hessian, barrier updater, and ccd."
            )
        except Exception as e:
            logger.error(f"Failed to update simulation state: {e}")

    def update_simulation_values(self, xtp1: Any) -> bool:
        try:
            self.params.vt = (xtp1 - self.params.xt) / self.params.dt
            self.params.xt = xtp1
            return True
        except Exception as e:
            logger.error(f"Failed to update simulation state: {e}")
            return False

    def simulation_step(self):
        """Perform a single simulation step."""
        if self.current_step >= self.max_steps:
            logger.info(f"Reached maximum simulation steps at step {self.current_step}.")
            self.shutdown()
            return

        if not self.simulation_state.get_attribute("running"):
            logger.info(f"Simulation is paused at step {self.current_step}. Waiting for commands...")
            time.sleep(1)
            return

        try:
            self.update_simulation_state(
                params=self.params,
                potential=self.potentials,
                gradient=self.gradient,
                hessian=self.hessian,
                barrier_updater=self.barrier_updater,
                ccd=self.ccd,
            )
        except Exception as e:
            logger.error(f"Failed to update simulation state at step {self.current_step}: {e}")
            return

        try:
            logger.info(f"Optimizing at step {self.current_step}...")
            xtp1 = self.optimizer.optimize(
                x0=self.params.xtilde,
                f=self.potentials,
                grad=self.gradient,
                hess=self.hessian,
                callback=self.barrier_updater,
            )
            self.update_simulation_values(xtp1)

            BX = to_surface(self.params.xt, self.params.mesh, self.params.cmesh)

            backend_data = {
                "timestamp": time.time(),
                "step": self.current_step,
                "positions": self.params.xt.tolist(),
                "BX": BX.tolist(),
            }

            db_data = {
                "timeStep": self.current_step,
                "timeElapsed": self.current_step * self.params.dt,
                "num_elements": len(self.simulation_state.get_attribute("num_nodes_list")),
                "positions": self.params.xt.reshape(-1, 3).tolist(),
                "velocity": self.params.vt.reshape(-1, 3).tolist(),
                "acceleration": self.params.a.reshape(-1, 3).tolist(),
            }

            # Send data update
            self.command_handler.send_data_update(backend_data, db_data)
            
        except Exception as e:
            logger.error(f"Error during simulation optimization at step {self.current_step}: {e}")
            return
        
        self.current_step += 1

    def run_simulation(self) -> None:
        """Starts the simulation by running the simulation loop."""
        try:
            logger.info("Starting simulation loop...")
            while self.current_step < self.max_steps:
                self.simulation_step()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user.")
            self.shutdown()
        except Exception as e:
            logger.error(f"An error occurred during simulation: {e}")
            self.shutdown()
        finally:
            logger.info("Simulation terminated gracefully.")
            self.shutdown()

    def shutdown(self):
        if self._shutdown_initiated:
            return 
        self._shutdown_initiated = True
        self.simulation_state.update_attribute("running", False)
        logger.info("Shutting down simulation...")
        try:
            if self.backend:
                self.backend.disconnect()
                logger.info("Backend disconnected successfully.")
        except Exception as e:
            logger.error(f"Error while disconnecting backend: {e}")
        

    def stop_simulation(self) -> None:
        """
        Stops the simulation gracefully.
        """
        logger.info("Stopping simulation gracefully...")
        self.shutdown()

    def pause_simulation(self) -> None:
        """
        Pauses the simulation.
        """
        logger.info("Pausing simulation...")
        self.simulation_state.update_attribute("running", False)
        logger.info("Simulation paused.")

    def resume_simulation(self) -> None:
        """
        Resumes the simulation.
        """
        logger.info("Resuming simulation...")
        self.simulation_state.update_attribute("running", True)
        logger.info("Simulation resumed.")

    def start_simulation(self) -> None:
        """
        Starts the simulation.
        """
        logger.info("Starting simulation...")
        self.simulation_state.update_attribute("running", True)
        logger.info("Simulation started.")
        self.run_simulation()
