import logging
import math
from pathlib import Path

import igl
import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp

from simulation.backend.factory import BackendFactory
from simulation.config.config import SimulationConfigManager
from simulation.core.modifier.mesh import compute_face_to_element_mapping, find_codim_vertices
from simulation.db.factory import DatabaseFactory
from simulation.io.io import combine_meshes, load_individual_meshes
from simulation.logs.error import (
    BackendInitializationError,
    BoundaryConditionsSetupError,
    CollisionSetupError,
    ConfigurationError,
    DatabaseInitializationError,
    ExternalForcesSetupError,
    HyperElasticSetupError,
    MassMatrixSetupError,
    SimulationErrorCode,
    StorageInitializationError,
)
from simulation.logs.message import SimulationLogMessageCode
from simulation.states.state import SimulationState
from simulation.storage.factory import StorageFactory

logger = logging.getLogger(__name__)


class SimulationInitializer:
    """Encapsulates the initialization process for the simulation."""

    def __init__(self, scenario: str = None):
        """
        Initializes the SimulationInitializer by setting up logging and loading the configuration.

        Args:
            scenario (str): Path to the scenario configuration file.
        """
        self.scenario = scenario
        self.simulation_state = None
        self.config_manager = None

        self.config_path = Path(scenario)
        self.config = None
        self.load_configuration()

    def load_configuration(self):
        logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"{self.config_path}"))
        try:
            self.config_manager = SimulationConfigManager(config_path=self.config_path)
            self.config = self.config_manager.get()
            logger.debug(SimulationLogMessageCode.CONFIGURATION_LOADED.details(f"{self.config}"))
        except Exception as e:
            logger.error(SimulationLogMessageCode.CONFIGURATION_FAILED.details(f"{e}"))
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def initialize_storage(self):
        logger.info(SimulationLogMessageCode.STORAGE_INITIALIZED)
        try:
            self.storage = StorageFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.STORAGE_FAILED.details(f"{e}"))
            raise StorageInitializationError(f"Failed to initialize storage: {e}")

    def initialize_backend(self):
        logger.info(SimulationLogMessageCode.BACKEND_INITIALIZED)
        try:
            self.backend = BackendFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.BACKEND_FAILED.details(f"{e}"))
            raise BackendInitializationError(f"Failed to initialize backend: {e}")

    def initialize_database(self):
        logger.info(SimulationLogMessageCode.DATABASE_INITIALIZED)
        try:
            self.database = DatabaseFactory.create(self.config_manager.get())
        except Exception as e:
            logger.error(SimulationLogMessageCode.DATABASE_FAILED.details(f"{e}"))
            raise DatabaseInitializationError(f"Failed to initialize database: {e}")

    def setup_initial_conditions(self, mesh: pbat.fem.Mesh):
        """Sets up the initial conditions for the simulation.

        Parameters
        ----------
        mesh : pbat.fem.Mesh
            The finite element mesh for the simulation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, int]
            - Position vector `x`.
            - Velocity vector `v`.
            - Acceleration vector `acceleration`.
            - Total number of nodes `n`.

        """
        x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
        n = x.shape[0]
        v = np.zeros(n)
        acceleration = np.zeros(n)
        logger.info(SimulationLogMessageCode.INITIAL_CONDITIONS_SET)
        return x, v, acceleration, n

    def setup_mass_matrix(self, mesh, materials, element_materials):
        logger.info(SimulationLogMessageCode.MASS_MATRIX_SETUP)
        try:
            # Retrieve and convert material densities
            densities = []
            for m in materials:
                density = m.get("density", {}).get("value", 1000.0)
                unit = m.get("density", {}).get("unit", "kg/m³")
                if unit == "g/cm³":
                    density *= 1000  # Convert g/cm³ to kg/m³
                densities.append(density)

            element_rho = np.array([densities[i] for i in element_materials])

            logger.info(
                SimulationLogMessageCode.MASS_MATRIX_SETUP.details(
                    f"Element densities: {element_rho}"
                )
            )

            # Create a 2D array with densities
            num_quadrature_points = 4
            rho = np.column_stack([element_rho] * num_quadrature_points)

            # Compute mass matrix
            M, detJeM = pbat.fem.mass_matrix(mesh, rho=7000, lump=True)
            Minv = sp.sparse.diags(1.0 / M.diagonal())
            return M, Minv
        except Exception as e:
            logger.error(SimulationLogMessageCode.MASS_MATRIX_FAILED.details(f"{e}"))
            raise MassMatrixSetupError(f"Failed to set up mass matrix: {e}")

    def setup_external_forces(self, mesh, materials, element_materials, Minv, gravity=9.81):
        logger.info(SimulationLogMessageCode.EXTERNAL_FORCES_SETUP)
        try:
            g = np.zeros(mesh.dims)
            g[-1] = -9.81
            f, detJeF = pbat.fem.load_vector(mesh, 7000 * g)
            a = Minv @ f
            return f, a, detJeF
        except Exception as e:
            logger.error(SimulationLogMessageCode.EXTERNAL_FORCES_FAILED.details(f"{e}"))
            raise ExternalForcesSetupError(f"Failed to set up external forces: {e}")

    def setup_hyperelastic_potential(self, mesh, materials, element_materials):
        logger.info(SimulationLogMessageCode.HYPERELASTIC_SETUP)
        try:
            detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
            GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)

            # Retrieve and convert Young's modulus values
            young_moduli = []
            for i in element_materials:
                young_modulus = materials[i].get("young_modulus", {}).get("value", 1e6)
                unit = materials[i].get("young_modulus", {}).get("unit", "Pa")
                if unit == "kPa":
                    young_modulus *= 1e3  # Convert kPa to Pa
                elif unit == "MPa":
                    young_modulus *= 1e6  # Convert MPa to Pa
                elif unit == "GPa":
                    young_modulus *= 1e9  # Convert GPa to Pa
                young_moduli.append(young_modulus)

            Y = np.array(young_moduli)
            nu = np.array([materials[i]["poisson"] for i in element_materials])

            psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
            hep, _, _, GNeU = pbat.fem.hyper_elastic_potential(mesh, Y=Y, nu=nu, energy=psi)
            return hep, Y, nu, psi, detJeU, GNeU
        except Exception as e:
            logger.error(SimulationLogMessageCode.HYPERELASTIC_FAILED.details(f"{e}"))
            raise HyperElasticSetupError(f"Failed to set up hyperelastic potential: {e}")

    def setup_collision_mesh(self, mesh, V, C, element_materials):
        logger.info(SimulationLogMessageCode.COLLISION_SETUP)
        try:
            boundary_faces = igl.boundary_facets(C)
            edges = ipctk.edges(boundary_faces)
            codim_vertices = find_codim_vertices(mesh, edges)

            if not codim_vertices:
                collision_mesh = ipctk.CollisionMesh.build_from_full_mesh(V, edges, boundary_faces)
            else:
                num_vertices = mesh.X.shape[1]
                is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(
                    num_vertices, edges, codim_vertices
                )
                collision_mesh = ipctk.CollisionMesh(is_on_surface, edges, boundary_faces)

            normal_collisions = ipctk.NormalCollisions()
            tangential_collisions = ipctk.TangentialCollisions()
            face_to_element_mapping = compute_face_to_element_mapping(C, boundary_faces)

            return (
                collision_mesh,
                edges,
                boundary_faces,
                normal_collisions,
                tangential_collisions,
                face_to_element_mapping,
            )
        except Exception as e:
            logger.error(SimulationLogMessageCode.COLLISION_FAILED.details(f"{e}"))
            raise CollisionSetupError(f"Failed to set up collision mesh: {e}")

    def setup_boundary_conditions(self, mesh, materials, num_nodes_list):
        logger.info(SimulationLogMessageCode.BOUNDARY_CONDITIONS_SETUP)
        dirichlet_bc_list = []
        node_offset = 0
        try:
            for _idx, (material, num_nodes) in enumerate(zip(materials, num_nodes_list)):
                percent_fixed = 0.0
                if percent_fixed > 0.0:
                    node_indices = slice(node_offset, node_offset + num_nodes)
                    X_sub = mesh.X[:, node_indices]
                    Xmin = X_sub.min(axis=1)
                    Xmax = X_sub.max(axis=1)
                    dX = Xmax - Xmin
                    fix_threshold = Xmin[-1] + percent_fixed * dX[-1]
                    fixed_nodes = (
                        np.where(mesh.X[-1, node_indices] <= fix_threshold)[0] + node_offset
                    )
                    if fixed_nodes.size > 0:
                        dirichlet_bc = np.repeat(fixed_nodes, mesh.dims) * mesh.dims + np.tile(
                            np.arange(mesh.dims), len(fixed_nodes)
                        )
                        dirichlet_bc_list.append(dirichlet_bc)
                node_offset += num_nodes

            if dirichlet_bc_list:
                all_dirichlet_bc = np.unique(np.concatenate(dirichlet_bc_list))
            else:
                all_dirichlet_bc = np.array([])

            total_dofs = np.setdiff1d(np.arange(mesh.X.shape[1] * mesh.dims), all_dirichlet_bc)
            return total_dofs, all_dirichlet_bc
        except Exception as e:
            logger.error(SimulationLogMessageCode.BOUNDARY_CONDITIONS_FAILED.details(f"{e}"))
            raise BoundaryConditionsSetupError(f"Failed to set up boundary conditions: {e}")

    def setup_collision_potentials(
        self, tangential_collisions, normal_collisions, collision_mesh, dhat, velocity
    ):
        ipctk.BarrierPotential.use_physical_barrier = True
        barrier_potential = ipctk.BarrierPotential(dhat)
        friction_potential = ipctk.FrictionPotential(velocity)
        return friction_potential, barrier_potential

    def initialize_simulation_state(
        self,
        config,
        mesh,
        x,
        v,
        acceleration,
        mass_matrix,
        hep,
        dt,
        collision_mesh,
        collision_constraints,
        friction_constraints,
        dhat,
        dmin,
        friction_coefficient,
        epsv,
        damping_coefficient,
        degrees_of_freedom,
        materials,
        barrier_potential,
        friction_potential,
        n,
        f_ext,
        qgf,
        rho_array,
        Y_array,
        nu_array,
        psi,
        detJeU,
        GNeU,
        element_materials,
        num_nodes_list,
        running=False,
        broad_phase_method=ipctk.BroadPhaseMethod.SWEEP_AND_PRUNE,
    ) -> SimulationState:
        """Initializes the simulation state with the given parameters.

        Parameters
        ----------
        All parameters represent attributes of the simulation state.

        Returns
        -------
        SimulationState
            The initialized simulation state with attributes and aliases set.

        """
        simulation_state = SimulationState()

        # Define attributes and their aliases
        attributes_with_aliases = {
            "config": (config, []),
            "mesh": (mesh, []),
            "x": (x, ["positions", "coordinates"]),
            "v": (v, ["velocities"]),
            "acceleration": (acceleration, ["a"]),
            "mass_matrix": (mass_matrix, ["M"]),
            "hep": (hep, ["hyper_elastic_potential"]),
            "dt": (dt, ["time_step", "delta_time"]),
            "cmesh": (collision_mesh, ["collision_mesh"]),
            "cconstraints": (collision_constraints, ["collision_constraints"]),
            "fconstraints": (friction_constraints, ["friction_constraints"]),
            "dhat": (dhat, ["distance_threshold"]),
            "dmin": (dmin, ["minimum_distance"]),
            "friction_coefficient": (friction_coefficient, ["mu"]),
            "epsv": (epsv, ["epsilon_v"]),
            "damping_coefficient": (damping_coefficient, []),
            "degrees_of_freedom": (degrees_of_freedom, ["dofs"]),
            "materials": (materials, []),
            "barrier_potential": (barrier_potential, []),
            "friction_potential": (friction_potential, []),
            "n": (n, ["normal_vector"]),
            "f_ext": (f_ext, ["external_force"]),
            "qgf": (qgf, ["qgf_potential"]),
            "rho_array": (rho_array, ["density_array"]),
            "Y_array": (Y_array, ["young_modulus_array"]),
            "nu_array": (nu_array, ["poisson_ratio_array"]),
            "psi": (psi, ["psi_energy"]),
            "detJeU": (detJeU, ["determinant_JeU"]),
            "GNeU": (GNeU, ["gradient_NeU"]),
            "element_materials": (element_materials, []),
            "num_nodes_list": (num_nodes_list, ["num_nodes"]),
            "running": (running, ["is_running"]),
            "broad_phase_method": (broad_phase_method, ["broad_phase"]),
        }

        # Add attributes to the simulation state
        for key, (value, aliases) in attributes_with_aliases.items():
            simulation_state.add_attribute(key, value, aliases=aliases)
            logger.info(f"Added attribute '{key}' to simulation state.")

        return simulation_state

    def extract_config_values(self):
        inputs = self.config.get("geometry", {}).get("meshes", [])
        friction_coefficient = self.config.get("contact", {}).get("friction", 0.0)
        damping_coefficient = self.config.get("contact", {}).get("damping_coefficient", 1e-4)
        dhat = self.config.get("contact", {}).get("dhat", 0.001)
        dmin = self.config.get("contact", {}).get("dmin", 0.0001)
        dt = self.config.get("time", {}).get("step", 0.1)
        gravity = self.config.get("time", {}).get("gravity", 9.81)
        epsv = self.config.get("contact", {}).get("epsv", 1e-6)

        return inputs, friction_coefficient, damping_coefficient, dhat, dmin, dt, gravity, epsv

    def initialize_simulation(self) -> SimulationState:
        """Executes the initialization steps for the simulation.

        Returns
        -------
        SimulationState
            The initialized simulation state.

        """
        try:

            # Extract configuration values using ConfigManager
            inputs, friction_coefficient, damping_coefficient, dhat, dmin, dt, gravity, epsv = (
                self.extract_config_values()
            )

            self.initialize_storage()
            self.initialize_backend()
            self.initialize_database()

            # Load input meshes and materials
            all_meshes, materials = load_individual_meshes(inputs, self.config_manager)
            logger.info(
                SimulationLogMessageCode.CONFIGURATION_LOADED.details(
                    f"Loaded {len(all_meshes)} meshes."
                )
            )

            # Combine all meshes into a single mesh
            mesh, V, C, element_materials, num_nodes_list = combine_meshes(all_meshes, materials)
            logger.info(SimulationLogMessageCode.CONFIGURATION_LOADED)

            # Setup initial conditions
            x, v, acceleration, n = self.setup_initial_conditions(mesh)

            # Setup mass matrix
            mass_matrix, inverse_mass_matrix = self.setup_mass_matrix(
                mesh, materials, element_materials
            )

            # Setup external forces
            f_ext, acceleration, qgf = self.setup_external_forces(
                mesh, materials, element_materials, inverse_mass_matrix, gravity
            )

            # Setup hyperelastic potential
            hep, Y_array, nu_array, psi, detJeU, GNeU = self.setup_hyperelastic_potential(
                mesh, materials, element_materials
            )

            # Setup collision mesh and constraints
            (
                collision_mesh,
                edges,
                boundary_faces,
                normal_collisions,
                tangential_collisions,
                face_to_element_mapping,
            ) = self.setup_collision_mesh(mesh, V, C, element_materials)

            # Setup boundary conditions
            degrees_of_freedom, dirichlet_boundary_conditions = self.setup_boundary_conditions(
                mesh, materials, num_nodes_list
            )

            # Setup collision potentials
            friction_potential, barrier_potential = self.setup_collision_potentials(
                tangential_collisions, normal_collisions, collision_mesh, dhat, epsv
            )

            rho_array = np.array(
                [materials[i].get("density", {}).get("value", 1000.0) for i in element_materials]
            )

            # Initialize SimulationState
            self.simulation_state = self.initialize_simulation_state(
                config=self.config,
                mesh=mesh,
                x=x,
                v=v,
                acceleration=acceleration,
                mass_matrix=mass_matrix,
                hep=hep,
                dt=dt,
                collision_mesh=collision_mesh,
                collision_constraints=normal_collisions,
                friction_constraints=tangential_collisions,
                dhat=dhat,
                dmin=dmin,
                friction_coefficient=friction_coefficient,
                epsv=epsv,
                damping_coefficient=damping_coefficient,
                degrees_of_freedom=degrees_of_freedom,
                materials=materials,
                barrier_potential=barrier_potential,
                friction_potential=friction_potential,
                n=gravity,
                f_ext=f_ext,
                qgf=qgf,
                rho_array=rho_array,
                Y_array=Y_array,
                nu_array=nu_array,
                psi=psi,
                detJeU=detJeU,
                GNeU=GNeU,
                element_materials=element_materials,
                num_nodes_list=num_nodes_list,
                running=False,
                broad_phase_method=ipctk.BroadPhaseMethod.SWEEP_AND_PRUNE,
            )

            logger.info(SimulationLogMessageCode.SIMULATION_STATE_INITIALIZED)

            return self.simulation_state

        except Exception as e:
            logger.error(SimulationLogMessageCode.SIMULATION_INITIALIZATION_ABORTED.details(f"{e}"))
            raise SimulationErrorCode(f"Failed to initialize simulation: {e}")
