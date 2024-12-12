# manager.py
import logging
from pathlib import Path
import sys
import time
from typing import Optional, Callable, Dict, Any

from simulation.backend.factory import BackendFactory
from simulation.config.config import SimulationConfigManager
from simulation.core.contact.barrier_updater import BarrierUpdaterFactory
from simulation.core.contact.ccd import CCDFactory
from simulation.core.math.gradient import GradientFactory
from simulation.core.math.hessian import HessianFactory
from simulation.core.math.potential import PotentialFactory
from simulation.core.modifier.mesh import to_surface
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.solvers.line_search import LineSearchFactory, LineSearchBase
from simulation.core.solvers.linear import LinearSolverFactory, LinearSolverBase
from simulation.core.solvers.optimizer import OptimizerFactory, OptimizerBase
from simulation.initializer import SimulationInitializer
from simulation.states.state import SimulationState
from simulation.controller.model import Request, Response, Status
from simulation.controller.dispatcher import CommandDispatcher
from simulation.storage.factory import StorageFactory
from simulation.logs.message import SimulationLogMessageCode
from simulation.logs.error import SimulationError, SimulationErrorCode

logger = logging.getLogger(__name__)

class SimulationManager:
    def __init__(self, scenario: str):
        self.scenario = scenario
        self.config_path = Path(scenario)
        self.config_manager = None
        self.config = None
        self.simulation_state = self.initialize_simulation(scenario)

        self.load_configuration()
        self.setup_factories()
        self.initialize_storage()
        self.initialize_backend()
        self.line_searcher = self.setup_line_searcher()
        self.linear_solver = self.setup_linear_solver()
        self.optimizer = self.setup_optimizer()

    def load_configuration(self):
        logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED)
        try:
            self.config_manager = SimulationConfigManager(config_path=self.config_path)
            self.config = self.config_manager.get()
            logger.debug(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"{self.config}"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(str(e)))
            raise SimulationError(SimulationErrorCode.CONFIGURATION, "Failed to load configuration", str(e))

    def initialize_storage(self):
        logger.info(SimulationLogMessageCode.STORAGE_INITIALIZED)
        try:
            self.storage = StorageFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.STORAGE_FAILED.details(str(e)))
            raise SimulationError(SimulationErrorCode.STORAGE_INITIALIZATION, "Failed to initialize storage", str(e))

    def initialize_backend(self):
        logger.info(SimulationLogMessageCode.BACKEND_INITIALIZED)
        try:
            self.backend = BackendFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.BACKEND_FAILED.details(str(e)))
            raise SimulationError(SimulationErrorCode.BACKEND_INITIALIZATION, "Failed to initialize backend", str(e))

    def setup_factories(self):
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()
        self.optimizer_factory = OptimizerFactory()
        self.storage_factory = StorageFactory()

    def initialize_simulation(self, scenario: str) -> SimulationState:
        logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
        try:
            simulation_state = SimulationInitializer(scenario=scenario).initialize_simulation()
            logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
            return simulation_state
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_INITIALIZATION_FAILED.details(str(e)))
            raise SimulationError(SimulationErrorCode.SIMULATION_LOOP, "Failed to initialize simulation", str(e))

    def reset_simulation(self) -> SimulationState:
        logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)
        return self.initialize_simulation(self.simulation_state.scenario)

    def setup_line_searcher(self) -> LineSearchBase:
        try:
            solver_config = self.config.get("solver", {}).get("optimizer", {})
            method = solver_config.get("line_search", "armijo").lower()
            if method not in ["armijo", "wolfe", "strong_wolfe", "parallel", "backtracking"]:
                logger.warning(SimulationLogMessageCode.PERFORMANCE_WARNING.details(f"Invalid line search method '{method}', falling back to 'armijo'"))
                method = "armijo"
            grad_required = method in ["wolfe", "strong_wolfe"]
            hep = self.simulation_state.get_attribute("hep")
            if hep is None:
                raise ValueError("Missing required objective function 'hep'")
            gradient = None
            if grad_required:
                gradient = self.simulation_state.get_attribute("gradient")
                if gradient is None:
                    raise ValueError(f"Gradient function required for {method} line search")
            line_searcher = self.line_search_factory.create(config=self.config, f=hep, grad_f=gradient)
            logger.info(SimulationLogMessageCode.POTENTIAL_SETUP.details(f"Line searcher '{method}' configured with gradient required: {grad_required}"))
            return line_searcher
        except Exception as e:
            logger.error(SimulationLogMessageCode.LINE_SEARCH_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.LINE_SEARCH_SETUP, "Failed to setup line searcher", str(e))

    def setup_linear_solver(self) -> LinearSolverBase:
        try:
            method = self.config.get("linear", {}).get("solver", "direct")
            method = "direct" if method == "default" else method
            dofs = self.simulation_state.get_attribute("degrees_of_freedom")
            linear_solver = self.linear_solver_factory.create(config=self.config, dofs=dofs)
            logger.info(SimulationLogMessageCode.POTENTIAL_SETUP.details(f"Linear solver '{method}' set up successfully."))
            return linear_solver
        except Exception as e:
            logger.error(SimulationLogMessageCode.LINEAR_SOLVER_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.LINEAR_SOLVER_SETUP, "Failed to setup linear solver", str(e))

    def setup_optimizer(self) -> OptimizerBase:
        try:
            optimizer_config = self.config.get("solver", {}).get("optimization", {})
            hep = self.simulation_state.get_attribute("hep")
            if hep is None:
                raise ValueError("Objective function 'hep' is missing in the simulation state.")
            gradient = self.simulation_state.get_attribute("gradient")
            linear_solver_config = self.config.get("solver", {}).get("linear", {})
            if not linear_solver_config:
                raise ValueError("Linear solver configuration is required.")
            dofs = self.simulation_state.get_attribute("degrees_of_freedom")
            if dofs is None:
                raise ValueError("Degrees of freedom (dofs) are missing in the simulation state.")
            optimizer = self.optimizer_factory.create(config=optimizer_config, f=hep, grad_f=gradient, linear_solver_config=linear_solver_config, dofs=dofs)
            logger.info(SimulationLogMessageCode.POTENTIAL_SETUP.details(f"Optimizer '{optimizer_config.get('solver', 'newton')}' set up successfully."))
            return optimizer
        except Exception as e:
            logger.error(SimulationLogMessageCode.OPTIMIZER_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.OPTIMIZER_SETUP, "Failed to setup optimizer", str(e))

    def setup_parameters(self) -> ParametersBase:
        logger.info(SimulationLogMessageCode.POTENTIAL_SETUP)
        required_attrs = [
            "mesh", "x", "v", "acceleration", "mass_matrix", "hep", "dt",
            "cmesh", "cconstraints", "fconstraints", "materials",
            "element_materials", "dhat", "dmin", "mu", "epsv",
            "barrier_potential", "friction_potential"
        ]
        attributes = {}
        for attr in required_attrs:
            attributes[attr] = self.simulation_state.get_attribute(attr, default=None)
        try:
            # Ensure the attributes match the Parameters class constructor
            parameters = Parameters(
                mesh=attributes["mesh"],
                xt=attributes["x"],  # Updated to match Parameters class constructor
                vt=attributes["v"],  # Updated to match Parameters class constructor
                a=attributes["acceleration"],  # Updated to match Parameters class constructor
                M=attributes["mass_matrix"],  # Updated to match Parameters class constructor
                hep=attributes["hep"],
                dt=attributes["dt"],
                cmesh=attributes["cmesh"],
                cconstraints=attributes["cconstraints"],
                fconstraints=attributes["fconstraints"],
                materials=attributes["materials"],
                element_materials=attributes["element_materials"],
                dhat=attributes["dhat"],
                dmin=attributes["dmin"],
                mu=attributes["mu"],
                epsv=attributes["epsv"],
                barrier_potential=attributes["barrier_potential"],
                friction_potential=attributes["friction_potential"]
            )
            logger.info(SimulationLogMessageCode.POTENTIAL_SETUP.details("Simulation parameters set up successfully."))
            return parameters
        except Exception as e:
            logger.error(SimulationLogMessageCode.PARAMETERS_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.PARAMETERS_SETUP, "Failed to initialize parameters", str(e))

    def setup_potentials(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.POTENTIAL_SETUP)
        try:
            potentials = PotentialFactory.create("default", params)
            logger.info(SimulationLogMessageCode.POTENTIAL_SETUP.details("Potentials set up successfully."))
            return potentials
        except Exception as e:
            logger.error(SimulationLogMessageCode.POTENTIAL_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.POTENTIAL_SETUP, "Failed to setup potentials", str(e))

    def setup_gradient(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.GRADIENT_SETUP)
        try:
            gradient = GradientFactory.create("default", params)
            logger.info(SimulationLogMessageCode.GRADIENT_SETUP.details("Gradient set up successfully."))
            return gradient
        except Exception as e:
            logger.error(SimulationLogMessageCode.GRADIENT_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.GRADIENT_SETUP, "Failed to setup gradient", str(e))

    def setup_hessian(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.HESSIAN_SETUP)
        try:
            hessian = HessianFactory.create("default", params)
            logger.info(SimulationLogMessageCode.HESSIAN_SETUP.details("Hessian set up successfully."))
            return hessian
        except Exception as e:
            logger.error(SimulationLogMessageCode.HESSIAN_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.HESSIAN_SETUP, "Failed to setup hessian", str(e))

    def setup_barrier_updater(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.BARRIER_UPDATER_SETUP)
        try:
            barrier_updater = BarrierUpdaterFactory.create("default", params)
            logger.info(SimulationLogMessageCode.BARRIER_UPDATER_SETUP.details("Barrier updater set up successfully."))
            return barrier_updater
        except Exception as e:
            logger.error(SimulationLogMessageCode.BARRIER_UPDATER_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.BARRIER_UPDATER_SETUP, "Failed to setup barrier updater", str(e))

    def setup_ccd(self, params: ParametersBase) -> Any:
        logger.info(SimulationLogMessageCode.CCD_SETUP)
        try:
            ccd = CCDFactory.create("default", params)
            logger.info(SimulationLogMessageCode.CCD_SETUP.details("CCD solver set up successfully."))
            return ccd
        except Exception as e:
            logger.error(SimulationLogMessageCode.CCD_SETUP.details(str(e)))
            raise SimulationError(SimulationErrorCode.CCD_SETUP, "Failed to setup CCD solver", str(e))

    def process_command(self, dispatcher: CommandDispatcher, request: Request) -> Optional[Response]:
        logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Processing command: {request.command_name}"))
        try:
            response = dispatcher.dispatch(request)
            if response:
                try:
                    self.backend.send_response(response)
                    logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Command {request.command_name} executed successfully"))
                except Exception as e:
                    logger.error(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Failed to send response: {e}"))
            return response
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Command processing failed: {e}"))
            return Response(request_id=request.request_id, status=Status.ERROR.value, message=f"Command processing failed: {str(e)}")

    def prepare_mesh_data(self, step: int) -> Optional[str]:
        logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Preparing mesh data for step {step}."))
        try:
            BX = to_surface(self.simulation_state.x, self.simulation_state.mesh, self.simulation_state.cmesh)
            mesh_data = {
                "timestamp": time.time(),
                "step": step,
                "x": self.simulation_state.x.tolist(),
                "BX": BX.tolist(),
                "faces": self.simulation_state.cmesh.faces.tolist(),
                "face_materials": self.simulation_state.face_materials.tolist(),
                "materials": [material.to_dict() for material in self.simulation_state.materials],
            }
            serialized = self.simulation_state.communication_client.serialize_data(mesh_data)
            if serialized:
                self.backend.write("mesh_state", serialized)
                self.backend.publish("simulation_updates", serialized)
                logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Step {step}: Published mesh data."))
                return serialized
            logger.warning(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Serialization failed at step {step}."))
            return None
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Failed to prepare mesh data at step {step}: {e}"))
            return None

    def handle_command(self, dispatcher: CommandDispatcher) -> None:
        request = self.backend.get_command()
        if not request:
            return
        logger.info(SimulationLogMessageCode.SIMULATION_STARTED.details("Command received."))
        try:
            response = self.process_command(dispatcher, request)
            metrics = dispatcher.get_metrics()
            logger.debug(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Command metrics: {metrics}"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Failed to process command: {e}"))

    def run_simulation(self) -> None:
        logger.info(SimulationLogMessageCode.SIMULATION_STARTED)
        params = self.setup_parameters()
        dispatcher = CommandDispatcher(self.simulation_state, reset_function=self.reset_simulation)
        max_iters = 100000

        for step in range(max_iters):
            self.simulation_state.update_attribute("running", True)
            self.handle_command(dispatcher)

            if not self.simulation_state.get_attribute("running"):
                logger.debug(SimulationLogMessageCode.SIMULATION_PAUSED.details("Simulation paused. Waiting for commands."))
                time.sleep(1)
                continue

            potentials = self.setup_potentials(params)
            gradient = self.setup_gradient(params)
            hessian = self.setup_hessian(params)
            barrier_updater = self.setup_barrier_updater(params)
            ccd = self.setup_ccd(params)

            self.update_simulation_state(params, potentials, gradient, hessian, barrier_updater, ccd)

            optimizer = self.setup_optimizer()
            try:
                xtp1 = optimizer.optimize(x0=params.xtilde, f=potentials, grad=gradient, hess=hessian, callback=barrier_updater)
                logger.debug(SimulationLogMessageCode.SIMULATION_STEP_COMPLETED.details(f"Optimization step {step} completed."))
            except Exception as e:
                logger.error(SimulationLogMessageCode.SIMULATION_STEP_COMPLETED.details(f"Optimization failed at step {step}: {e}"))
                continue

            if not self.update_simulation_values(params, xtp1, step):
                continue

            self.prepare_mesh_data(step)

        logger.info(SimulationLogMessageCode.SIMULATION_SHUTDOWN)
        self.simulation_state.update_attribute("running", False)

    def update_simulation_state(self, params: ParametersBase, potential: Any, gradient: Any, hessian: Any, barrier_updater: Any, ccd: Any) -> None:
        self.simulation_state.update_attributes({
            "params": params,
            "potential": potential,
            "gradient": gradient,
            "hessian": hessian,
            "barrier_updater": barrier_updater,
            "ccd": ccd,
        })
        logger.debug(SimulationLogMessageCode.SIMULATION_STARTED.details("Simulation state updated."))

    def update_simulation_values(self, params: ParametersBase, xtp1: Any, step: int) -> bool:
        try:
            params.vt = (xtp1 - params.xt) / params.dt
            params.xt = xtp1
            params.xtilde = params.xt + params.dt * params.vt + (params.dt ** 2 * params.acceleration)
            logger.debug(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Updated simulation state at step {step}."))
            return True
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_STARTED.details(f"Failed to update simulation state at step {step}: {e}"))
            return False

    def shutdown(self) -> None:
        logger.info(SimulationLogMessageCode.SIMULATION_SHUTDOWN)
        try:
            if hasattr(self, 'backend') and self.backend:
                self.backend.disconnect()
                logger.info(SimulationLogMessageCode.SIMULATION_SHUTDOWN.details("Backend closed successfully."))
        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_SHUTDOWN.details(f"Error while closing backend: {e}"))
        sys.exit(0)