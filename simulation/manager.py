# manager.py
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simulation.backend.factory import BackendFactory
from simulation.config.config import SimulationConfigManager
from simulation.controller.commands import GetBackendStatusCommand, SendDataCommand, UpdateParameterCommand
from simulation.controller.dispatcher import CommandDispatcher
from simulation.controller.factory import CommandFactory
from simulation.controller.model import Request, Response, Status
from simulation.core.contact.barrier import BarrierFactory
from simulation.core.contact.ccd import CCDFactory
from simulation.core.math.gradient import GradientFactory
from simulation.core.math.hessian import HessianFactory
from simulation.core.math.potential import PotentialFactory
from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.solvers.line_search import LineSearchBase, LineSearchFactory
from simulation.core.solvers.linear import LinearSolverBase, LinearSolverFactory
from simulation.core.solvers.optimizer import OptimizerBase, OptimizerFactory
from simulation.initializer import SimulationInitializer
from simulation.logs.error import SimulationError, SimulationErrorCode
from simulation.logs.message import SimulationLogMessageCode
from simulation.states.state import SimulationState
from simulation.storage.factory import StorageFactory

logger = logging.getLogger(__name__)


class SimulationManager:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.config_path = Path(scenario)
        self.config_manager = None
        self.config = None
        self.simulation_state = self.initialize_simulation(scenario)
        self.backend = None
        self.storage = None
        self.command_factory = None

        self.load_configuration()
        self.setup_factories()
        self.initialize_storage()
        self.initialize_backend()
        self.line_searcher = self.setup_line_searcher()
        self.linear_solver = self.setup_linear_solver()
        self.optimizer = self.setup_optimizer()

        self.total_time = self.config.get("time", {}).get("total", 1.0)
        self.time_step = self.config.get("time", {}).get("step", 0.01)
        self.max_steps = int(self.total_time / self.time_step)

        self.setup_commands()

    def load_configuration(self):
        logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED)
        try:
            self.config_manager = SimulationConfigManager(config_path=self.config_path)
            self.config = self.config_manager.get()
            logger.debug(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"{self.config}"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.CONFIGURATION, "Failed to load configuration", str(e)
            )

    def initialize_storage(self):
        logger.info(SimulationLogMessageCode.STORAGE_INITIALIZED)
        try:
            self.storage = StorageFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.STORAGE_FAILED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.STORAGE_INITIALIZATION, "Failed to initialize storage", str(e)
            )

    def initialize_backend(self):
        logger.info(SimulationLogMessageCode.BACKEND_INITIALIZED)
        try:
            self.backend = BackendFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.BACKEND_FAILED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.BACKEND_INITIALIZATION, "Failed to initialize backend", str(e)
            )

    def setup_factories(self):
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()
        self.optimizer_factory = OptimizerFactory()
        self.storage_factory = StorageFactory()

    def setup_commands(self):
        """Set up commands within the simulation."""
        self.command_factory = CommandFactory(history=his
        try:
            # Register commands with the CommandFactory
            CommandFactory.register("send_data", SendDataCommand)
            CommandFactory.register("update_parameter", UpdateParameterCommand)
            CommandFactory.register("get_backend_status", GetBackendStatusCommand)
            logger.info(SimulationLogMessageCode.COMMANDS_REGISTERED.details("Commands initialized successfully"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.COMMANDS_FAILED.details(str(e)))
            raise SimulationError(SimulationErrorCode.COMMAND_REGISTRATION, "Failed to register commands", str(e))


    def initialize_simulation(self, scenario: str) -> SimulationState:
        logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
        try:
            simulation_state = SimulationInitializer(scenario=scenario).initialize_simulation()
            logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
            return simulation_state
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_INITIALIZATION_ABORTED.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.SIMULATION_LOOP, "Failed to initialize simulation", str(e)
            )

    def reset_simulation(self) -> SimulationState:
        logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
        return self.initialize_simulation(self.simulation_state.scenario)

    def setup_line_searcher(self) -> LineSearchBase:
        try:
            solver_config = self.config.get("solver", {}).get("optimizer", {})
            method = solver_config.get("line_search", "armijo").lower()

            # Validate method early
            valid_methods = {"armijo", "wolfe", "strong_wolfe", "parallel", "backtracking"}
            if method not in valid_methods:
                logger.warning(f"Invalid line search method '{method}', falling back to 'armijo'")
                method = "armijo"

            # Clear gradient requirements
            grad_required = method in {"wolfe", "strong_wolfe"}

            # Get and validate objective function
            hep = self.simulation_state.get_attribute("hep")
            if hep is None:
                raise ValueError("Missing required objective function 'hep'")

            # Get and validate gradient if required
            gradient = None
            if grad_required:
                gradient = self.simulation_state.get_attribute("gradient")
                if gradient is None:
                    raise ValueError(f"Gradient function required for {method} line search")

            # Create line searcher with validated components
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

    def setup_parameters(self) -> ParametersBase:
        logger.info(SimulationLogMessageCode.POTENTIAL_SETUP)
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
            # Create the Parameters object
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
            logger.info(
                SimulationLogMessageCode.POTENTIAL_SETUP.details(
                    "Simulation parameters set up successfully."
                )
            )
            return parameters

        except Exception as e:
            logger.error(SimulationLogMessageCode.PARAMETERS_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.PARAMETERS_SETUP, "Failed to initialize parameters", str(e)
            )

    def setup_potentials(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.POTENTIAL_SETUP)
        try:
            potentials = PotentialFactory.create(params=params)
            logger.info(
                SimulationLogMessageCode.POTENTIAL_SETUP.details("Potentials set up successfully.")
            )
            return potentials
        except Exception as e:
            logger.error(SimulationLogMessageCode.POTENTIAL_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.POTENTIAL_SETUP, "Failed to setup potentials", str(e)
            )

    def setup_gradient(self, params: ParametersBase) -> Any:
        """Setup gradient with parameter validation."""
        logger.info(SimulationLogMessageCode.GRADIENT_SETUP)
        try:
            # Validate parameters
            if params is None:
                # Try to get params from simulation state
                params = self.simulation_state.get_attribute("params")
                if params is None:
                    # Create new parameters if none exist
                    params = self.setup_parameters()
                    self.simulation_state.update_attribute("params", params)

            # Validate required parameters
            required_params = [
                "mesh",
                "dt",
                "xt",
                "vt",
                "M",
                "hep",
                "cmesh",
                "materials",
                "element_materials",
            ]

            missing_params = [
                param
                for param in required_params
                if not hasattr(params, param) or getattr(params, param) is None
            ]

            if missing_params:
                raise ValueError(
                    f"Missing required parameters for gradient: {', '.join(missing_params)}"
                )

            # Create gradient with validated parameters
            gradient = GradientFactory.create(params=params)

            # Test gradient with dummy input
            test_x = np.zeros(len(params.xt))
            try:
                test_grad = gradient(test_x)
                if not isinstance(test_grad, np.ndarray):
                    raise ValueError("Gradient must return numpy array")
            except Exception as e:
                raise ValueError(f"Invalid gradient function: {str(e)}")
            
            self.update_params("gradient", gradient)

            logger.info(
                SimulationLogMessageCode.GRADIENT_SETUP.details("Gradient setup successful")
            )
            return gradient

        except Exception as e:
            logger.error(SimulationLogMessageCode.GRADIENT_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.GRADIENT_SETUP, "Failed to setup gradient", str(e)
            )

    def setup_hessian(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.HESSIAN_SETUP)
        try:
            hessian = HessianFactory.create(params=params)
            logger.info(
                SimulationLogMessageCode.HESSIAN_SETUP.details("Hessian set up successfully.")
            )

            self.update_params("hessian", hessian)

            return hessian
        except Exception as e:
            logger.error(SimulationLogMessageCode.HESSIAN_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.HESSIAN_SETUP, "Failed to setup hessian", str(e)
            )

    def setup_barrier_updater(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.BARRIER_UPDATER_SETUP)
        try:
            barrier_updater = BarrierFactory.create_updater(params=params)
            logger.info(
                SimulationLogMessageCode.BARRIER_UPDATER_SETUP.details(
                    "Barrier updater set up successfully."
                )
            )
            self.update_params("barrier_updater", barrier_updater)
            return barrier_updater
        except Exception as e:
            logger.error(SimulationLogMessageCode.BARRIER_UPDATER_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.BARRIER_UPDATER_SETUP, "Failed to setup barrier updater", str(e)
            )

    def setup_ccd(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.CCD_SETUP)
        try:
            ccd = CCDFactory.create(params)
            logger.info(
                SimulationLogMessageCode.CCD_SETUP.details("CCD solver set up successfully.")
            )
            self.update_params("ccd", ccd)
            return ccd
        except Exception as e:
            logger.error(SimulationLogMessageCode.CCD_SETUP.details(str(e)))
            raise SimulationError(
                SimulationErrorCode.CCD_SETUP, "Failed to setup CCD solver", str(e)
            )

    def process_command(
        self, dispatcher: CommandDispatcher, request: Request
    ) -> Optional[Response]:
        logger.info(
            SimulationLogMessageCode.SIMULATION_STARTED.details(
                f"Processing command: {request.command_name}"
            )
        )
        try:
            response = dispatcher.dispatch(request)
            if response:
                try:
                    self.backend.send_response(response)
                    logger.info(
                        SimulationLogMessageCode.SIMULATION_STARTED.details(
                            f"Command {request.command_name} executed successfully"
                        )
                    )
                except Exception as e:
                    logger.error(
                        SimulationLogMessageCode.SIMULATION_STARTED.details(
                            f"Failed to send response: {e}"
                        )
                    )
            return response
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.SIMULATION_STARTED.details(
                    f"Command processing failed: {e}"
                )
            )
            return Response(
                request_id=request.request_id,
                status=Status.ERROR.value,
                message=f"Command processing failed: {str(e)}",
            )
        
    def process_command(self, dispatcher: CommandDispatcher, request: Request) -> Optional[Response]:
        logger.info(
            SimulationLogMessageCode.COMMAND_RECEIVED.details(f"Processing command: {request.command_name}")
        )
        try:
            response = dispatcher.dispatch(request)
            if response:
                try:
                    self.backend.send_response(response)
                    logger.info(
                        SimulationLogMessageCode.COMMAND_EXECUTED.details(f"Command {request.command_name} executed successfully")
                    )
                except Exception as e:
                    logger.error(
                        SimulationLogMessageCode.COMMAND_EXECUTION_FAILED.details(f"Failed to send response: {e}")
                    )
            return response
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_PROCESSING_FAILED.details(f"Command processing failed: {e}")
            )
            return Response(
                request_id=request.request_id,
                status=Status.ERROR.value,
                message=f"Command processing failed: {str(e)}",
            )

    def handle_command(self, dispatcher: CommandDispatcher) -> None:
        # Get the command from the backend
        request = self.backend.get_command()

        if not request:
            return

        logger.info(SimulationLogMessageCode.COMMAND_RECEIVED.details(f"Command received: {request.command_name}"))

        try:
            # If the command is 'start', begin the simulation process
            if request.command_name == "start":
                logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details("Simulation starting..."))
                self.simulation_state.update_attribute("running", True)
                self.process_command(dispatcher, request)  # Process the start command
            elif self.simulation_state.get_attribute("running"):
                # If simulation is running, process other commands
                response = self.process_command(dispatcher, request)
                if response:
                    metrics = dispatcher.get_metrics()
                    logger.debug(
                        SimulationLogMessageCode.COMMAND_METRICS.details(f"Command metrics: {metrics}")
                    )

        except Exception as e:
            logger.error(
                SimulationLogMessageCode.COMMAND_EXECUTION_FAILED.details(f"Failed to process command: {e}")
            )

    def run_simulation(self) -> None:
        logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details("Waiting for 'start' command..."))
        
        # Wait for the start command before beginning the simulation
        while not self.simulation_state.get_attribute("running"):
            time.sleep(1)  # Check periodically for the start command

        logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details("Simulation started."))
        dispatcher = CommandDispatcher()
        params = self.setup_parameters()

        for step in range(self.max_steps):
            self.simulation_state.update_attribute("running", True)
            self.handle_command(dispatcher)  # Handle any incoming commands during the simulation

            if not self.simulation_state.get_attribute("running"):
                logger.debug(
                    SimulationLogMessageCode.SIMULATION_PAUSED.details(
                        "Simulation paused. Waiting for commands."
                    )
                )
                time.sleep(1)
                continue

            # Simulation logic (e.g., optimization steps)
            potentials = self.setup_potentials(params)
            gradient = self.setup_gradient(params)
            hessian = self.setup_hessian(params)
            barrier_updater = self.setup_barrier_updater(params)
            ccd = self.setup_ccd(params)

            self.update_simulation_state(
                params, potentials, gradient, hessian, barrier_updater, ccd
            )

            optimizer = self.setup_optimizer()
            try:
                logger.info(f"Optimizing at step {step}...")
                xtp1 = optimizer.optimize(
                    x0=params.xtilde,
                    f=potentials,
                    grad=gradient,
                    hess=hessian,
                    callback=barrier_updater,
                )
                logger.debug(
                    SimulationLogMessageCode.SIMULATION_STEP_COMPLETED.details(
                        f"Optimization step {step} completed."
                    )
                )
            except Exception as e:
                logger.error(
                    SimulationLogMessageCode.SIMULATION_STEP_FAILED.details(
                        f"Optimization failed at step {step}: {e}"
                    )
                )
                continue

            self.update_simulation_values(params, xtp1, step)

        logger.info(SimulationLogMessageCode.SIMULATION_SHUTDOWN.details("Simulation shutting down..."))
        self.simulation_state.update_attribute("running", False)

    def update_simulation_state(
        self,
        params: ParametersBase,
        potential: Any,
        gradient: Any,
        hessian: Any,
        barrier_updater: Any,
        ccd: Any,
    ) -> None:
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
        # show the simulation state
        logger.info(f"Simulation state: {self.simulation_state}")
        logger.debug(
            SimulationLogMessageCode.SIMULATION_STARTED.details("Simulation state updated.")
        )

    def update_params(self, key: str, value: Any) -> None:
        """
        Update the params attribute in SimulationState and log the changes.
        """
        params = self.simulation_state.get_attribute("params")
        if params is None:
            params = self.setup_parameters()
            self.simulation_state.update_attribute("params", params)
        setattr(params, key, value)
        logger.debug(f"Updated params[{key}] with new value.")

    def update_simulation_values(self, params: ParametersBase, xtp1: Any, step: int) -> bool:
        try:
            params.vt = (xtp1 - params.xt) / params.dt

            params.xtilde = xtp1 + params.dt * params.vt + params.dt2 * params.a

            params.xt = xtp1

            logger.debug(
                SimulationLogMessageCode.SIMULATION_STARTED.details(
                    f"Updated simulation state at step {step}."
                )
            )
            return True
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.SIMULATION_STARTED.details(
                    f"Failed to update simulation state at step {step}: {e}"
                )
            )
            return False

    def shutdown(self) -> None:
        logger.info(SimulationLogMessageCode.SIMULATION_SHUTDOWN)
        try:
            if hasattr(self, "backend") and self.backend:
                self.backend.disconnect()
                logger.info(
                    SimulationLogMessageCode.SIMULATION_SHUTDOWN.details(
                        "Backend closed successfully."
                    )
                )
        except Exception as e:
            logger.error(
                SimulationLogMessageCode.SIMULATION_SHUTDOWN.details(
                    f"Error while closing backend: {e}"
                )
            )
        sys.exit(0)
