import logging
import sys
import time
from typing import Optional

from simulation.core.contact.barrier_updater import BarrierUpdater, BarrierUpdaterFactory
from simulation.core.contact.ccd import CCDBase, CCDFactory
from simulation.core.math.gradient import GradientBase, GradientFactory
from simulation.core.math.hessian import HessianBase, HessianFactory
from simulation.core.math.potential import PotentialBase, PotentialFactory
from simulation.core.parameters import Parameters, ParametersBase
from simulation.core.solvers.line_search import LineSearchBase, LineSearchFactory
from simulation.core.solvers.linear import LinearSolverBase, LinearSolverFactory
from simulation.core.solvers.optimizer import OptimizerBase, OptimizerFactory
from simulation.core.modifier.mesh import to_surface
from simulation.init import SimulationInitializer
from simulation.nets.controller.factory import CommandDispatcher
from simulation.nets.messages import RequestMessage, ResponseMessage

logger = logging.getLogger(__name__)


class SimulationManager:
    def __init__(self):
        self.logger = logger
        self.simulation_state = self.initialize_simulation()

        # Add connection factories
        self.network_factory = NetsFactory()
        self.storage_factory = StorageFactory()
        self.database_factory = DatabaseFactory()

        # Existing factories
        self.line_search_factory = LineSearchFactory()
        self.linear_solver_factory = LinearSolverFactory()
        self.optimizer_factory = OptimizerFactory()

        self.line_searcher = self.setup_line_searcher()
        self.linear_solver = self.setup_linear_solver()

    def initialize_simulation(self) -> SimulationState:
        """Initializes the simulation by creating a new SimulationState object."""
        self.logger.info("Initializing simulation...")
        self.simulation_state = SimulationInitializer().initialize_simulation()
        self.logger.info("Simulation initialized successfully.")
        return self.simulation_state

    def reset_simulation(self) -> SimulationState:
        """Resets the simulation by re-initializing the SimulationState."""
        return self.initialize_simulation()

    def setup_line_searcher(self) -> LineSearchBase:
        """Sets up the LineSearch instance using the factory."""
        config = self.simulation_state.get_attribute("config")
        line_search_method = config.get("line_search_method", "backtracking").lower()
        grad_f_required = line_search_method in ["wolfe", "strong_wolfe"]

        line_searcher = self.line_search_factory.create(
            type=line_search_method,
            f=self.simulation_state.get_attribute("hep"),
            grad_f=(self.simulation_state.get_attribute("gradient") if grad_f_required else None),
            maxiters=config.get("line_search_maxiters", 20),
            c=config.get("c", 1e-4),
            tau=config.get("tau", 0.5),
            c1=config.get("c1", 1e-4),
            c2=config.get("c2", 0.9),
        )
        self.logger.info(f"Line searcher '{line_search_method}' set up successfully.")
        return line_searcher

    def setup_linear_solver(self) -> LinearSolverBase:
        """Sets up the LinearSolver instance using the factory."""
        config = self.simulation_state.get_attribute("config")
        linear_solver_method = config.get("linear_solver", "direct").lower()
        if linear_solver_method == "default":
            linear_solver_method = "direct"  # Map 'default' to a valid solver type

        dofs = self.simulation_state.get_attribute("degrees_of_freedom")
        linear_solver = self.linear_solver_factory.create(
            type=linear_solver_method,
            dofs=dofs,
            reg_param=config.get("reg_param", 1e-4),
        )
        self.logger.info(f"Linear solver '{linear_solver_method}' set up successfully.")
        return linear_solver

    def setup_optimizer(self) -> OptimizerBase:
        """Sets up the Optimizer instance using the factory."""
        config = self.simulation_state.get_attribute("config")
        optimizer_type = config.get("optimizer", "default").lower()
        ccd = self.simulation_state.get_attribute("ccd")
        dofs = self.simulation_state.get_attribute("degrees_of_freedom")

        optimizer = self.optimizer_factory.create(
            type=optimizer_type,
            line_searcher=self.line_searcher,
            alpha0_func=ccd,
            lsolver=self.linear_solver if optimizer_type == "newton" else None,
            maxiters=config.get("maxiters", 100),
            rtol=config.get("rtol", 1e-5),
            n_threads=config.get("n_threads", 1),
            reg_param=config.get("reg_param", 1e-4),
            m=config.get("m", 10),
            dofs=dofs,
        )
        self.logger.info(f"Optimizer '{optimizer_type}' set up successfully.")
        return optimizer

    def setup_parameters(self) -> ParametersBase:
        self.logger.info("Setting up simulation parameters.")
        self.simulation_state.check_required_attributes(
            [
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
                "broad_phase_method",
            ]
        )

        return Parameters(
            mesh=self.simulation_state.get_attribute("mesh"),
            xt=self.simulation_state.get_attribute("x"),
            vt=self.simulation_state.get_attribute("v"),
            a=self.simulation_state.get_attribute("acceleration"),
            M=self.simulation_state.get_attribute("mass_matrix"),
            hep=self.simulation_state.get_attribute("hep"),
            dt=self.simulation_state.get_attribute("dt"),
            cmesh=self.simulation_state.get_attribute("cmesh"),
            cconstraints=self.simulation_state.get_attribute("cconstraints"),
            fconstraints=self.simulation_state.get_attribute("fconstraints"),
            materials=self.simulation_state.get_attribute("materials"),
            element_materials=self.simulation_state.get_attribute("element_materials"),
            dhat=self.simulation_state.get_attribute("dhat"),
            dmin=self.simulation_state.get_attribute("dmin"),
            mu=self.simulation_state.get_attribute("mu"),
            epsv=self.simulation_state.get_attribute("epsv"),
            barrier_potential=self.simulation_state.get_attribute("barrier_potential"),
            friction_potential=self.simulation_state.get_attribute("friction_potential"),
            broad_phase_method=self.simulation_state.get_attribute("broad_phase_method"),
        )

    def setup_potentials(self, params: ParametersBase) -> PotentialBase:
        self.logger.info("Setting up potentials.")
        return PotentialFactory.create("default", params)

    def setup_gradient(self, params: ParametersBase) -> GradientBase:
        self.logger.info("Setting up gradient.")
        return GradientFactory.create("default", params)

    def setup_hessian(self, params: ParametersBase) -> HessianBase:
        self.logger.info("Setting up Hessian.")
        return HessianFactory.create("default", params)

    def setup_barrier_updater(self, params: ParametersBase) -> BarrierUpdater:
        self.logger.info("Setting up barrier updater.")
        return BarrierUpdaterFactory.create("default", params)

    def setup_ccd(self, params: ParametersBase) -> CCDBase:
        self.logger.info("Setting up CCD solver.")
        return CCDFactory.create("default", params)

    def process_command(
        self, dispatcher: CommandDispatcher, request: RequestMessage
    ) -> Optional[ResponseMessage]:
        self.logger.info(f"Processing command: {request.command}")
        response = dispatcher.dispatch(request)
        if response:
            dispatcher.simulation_state.communication_client.send_response(response)
        return response

    def prepare_mesh_data(self, step: int) -> Optional[str]:
        self.logger.info(f"Preparing mesh data for step {step}.")
        try:
            BX = to_surface(
                self.simulation_state.x,
                self.simulation_state.mesh,
                self.simulation_state.cmesh,
            )

            mesh_data = {
                "timestamp": time.time(),
                "step": step,
                "x": self.simulation_state.x.tolist(),
                "BX": BX.tolist(),
                "faces": self.simulation_state.cmesh.faces.tolist(),
                "face_materials": self.simulation_state.face_materials.tolist(),
                "materials": [material.to_dict() for material in self.simulation_state.materials],
            }

            mesh_data_serialized = self.simulation_state.communication_client.serialize_data(
                mesh_data
            )

            if mesh_data_serialized:
                self.simulation_state.communication_client.set_data(
                    "mesh_state", mesh_data_serialized
                )
                self.simulation_state.communication_client.publish_data(
                    "simulation_updates", mesh_data_serialized
                )
                self.logger.info(
                    f"Step {step}: Published mesh data to 'simulation_updates' channel."
                )
                return mesh_data_serialized
            self.logger.warning(f"Serialization failed at step {step}. Mesh data not published.")
            return None
        except Exception as e:
            self.logger.error(f"Failed to prepare mesh data at step {step}: {e}")
            return None

    def handle_command(self, dispatcher: CommandDispatcher) -> None:
        # Check for incoming commands
        request_data = self.simulation_state.get_attribute("communication_client").get_command()
        if request_data:
            try:
                if isinstance(request_data, dict):
                    request = RequestMessage(**request_data)
                elif isinstance(request_data, RequestMessage):
                    request = request_data
                else:
                    raise ValueError("Invalid request format.")
                self.logger.info(f"Processing command: {request.command}")
                self.process_command(dispatcher, request)
            except Exception as e:
                self.logger.error(f"Failed to process command: {e}")

    def run_simulation(self) -> None:
        self.logger.info("Initializing simulation setup.")
        params = self.setup_parameters()

        dispatcher = CommandDispatcher(self.simulation_state, reset_function=self.reset_simulation)

        self.logger.info("Starting simulation loop.")
        max_iters = 100000

        for step in range(max_iters):
            # Update the 'running' status based on simulation state
            self.simulation_state.update_attribute("running", True)

            # Handle incoming commands
            self.handle_command(dispatcher)

            # Check if simulation is running
            if not self.simulation_state.get_attribute("running"):
                self.logger.debug("Simulation is paused or not started. Waiting for commands.")
                time.sleep(1)
                continue

            potential = self.setup_potentials(params)
            gradient = self.setup_gradient(params)
            hessian = self.setup_hessian(params)
            barrier_updater = self.setup_barrier_updater(params)
            ccd = self.setup_ccd(params)

            self.simulation_state.update_attribute("params", params)
            self.simulation_state.update_attribute("potential", potential)
            self.simulation_state.update_attribute("gradient", gradient)
            self.simulation_state.update_attribute("hessian", hessian)
            self.simulation_state.update_attribute("barrier_updater", barrier_updater)
            self.simulation_state.update_attribute("ccd", ccd)

            optimizer = self.setup_optimizer()

            # Perform optimization step
            try:
                xtp1 = optimizer.optimize(
                    x0=params.xtilde,
                    f=potential,
                    grad=gradient,
                    hess=hessian,
                    callback=barrier_updater,
                )
                self.logger.debug(f"Optimization step {step} completed.")
            except Exception as e:
                self.logger.error(f"Optimization failed at step {step}: {e}")
                continue

            # Update simulation state
            try:
                params.vt = (xtp1 - params.xt) / params.dt
                params.xt = xtp1
                params.xtilde = params.xt + params.dt * params.vt + (params.dt**2 * params.a)
                self.logger.debug(f"Updated simulation state at step {step}.")
            except Exception as e:
                self.logger.error(f"Failed to update simulation state at step {step}: {e}")
                continue

            # Prepare mesh data and publish updates
            self.prepare_mesh_data(step)

        self.logger.info("Simulation loop ended.")
        self.simulation_state.update_attribute("running", False)

    def shutdown(self) -> None:
        """Shuts down the simulation."""
        self.logger.info("Shutting down simulation.")
        try:
            communication_client = self.simulation_state.get_attribute("communication_client")
            if communication_client:
                communication_client.close()
                self.logger.info("Communication client closed successfully.")
        except Exception as e:
            self.logger.error(f"Error while closing communication client: {e}")
        sys.exit(0)
