import argparse
import logging
import math
import sys
from pathlib import Path

import igl
import ipctk
import numpy as np
import pbatoolkit as pbat
import scipy as sp

from simulation.config.config import ConfigManager
from simulation.states.state import SimulationState
from simulation.io.io import combine_meshes, load_individual_meshes
from simulation.core.modifier.mesh import compute_face_to_element_mapping, find_codim_vertices
from simulation.nets.factory import (
    NetsFactory,
)
from simulation.db.factory import DatabaseFactory

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="3D Elastic Simulation of Linear FEM Tetrahedra using IPC",
        description="Simulate 3D elastic deformations with contact handling.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path to JSON file with simulation parameters",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to YAML file with simulation parameters",
    )
    return parser.parse_args()


class SimulationInitializer:
    """Encapsulates the initialization process for the simulation."""

    def __init__(self):
        """Initializes the SimulationInitializer by setting up logging and loading the configuration."""
        try:
            # Parse command-line arguments
            self.args = parse_arguments()

            # Initialize ConfigManager singleton
            self.config_manager = ConfigManager.get_instance()

            # Determine configuration file path
            config_file_path = None
            if self.args.json:
                config_file_path = self.args.json
                self.logger.debug(
                    f"Loading configuration from JSON file: {config_file_path}"
                )
            elif self.args.yaml:
                config_file_path = self.args.yaml
                self.logger.debug(
                    f"Loading configuration from YAML file: {config_file_path}"
                )

            # Initialize ConfigManager with the configuration file
            self.config_manager.initialize(
                config_name=Path(config_file_path).name if config_file_path else "config.yaml",
                config_path=str(Path(config_file_path).parent) if config_file_path else "./configs"
            )

            # Apply logging configuration from ConfigManager
            logging_config = self.config_manager.get_param("logging", {})
            self.logging_manager.apply_config(logging_config)
            self.logger.info("Logging has been initialized.")

            # Load simulation configuration using ConfigManager
            self.config = self.config_manager.config  # Access the OmegaConf DictConfig directly
            self.logger.info("Simulation configuration loaded successfully.")
            self.logger.debug(f"Configuration details: {self.config}")

            self.simulation_state = None

            # Add connection factories
            self.network_factory = NetworkConnectionFactory()
            self.storage_factory = StorageConnectionFactory()
            self.database_factory = DatabaseConnectionFactory()

        except Exception as e:
            logging.error(f"Failed to initialize SimulationInitializer: {e}")
            sys.exit(1)

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
        logger.info("Initial conditions set up.")
        return x, v, acceleration, n

    def setup_mass_matrix(self, mesh, materials, element_materials):
        self.logger.info("Setting up mass matrix.")
        try:
            # Example: Retrieve mass density from materials if needed
            rho = self.config_manager.get_param("materials", [{}])[0].get("density", 1000.0)
            M, detJe = pbat.fem.mass_matrix(mesh, rho=rho, lump=True)
            Minv = sp.sparse.diags(1.0 / M.diagonal())
            return M, Minv
        except Exception as e:
            self.logger.error(f"Failed to set up mass matrix: {e}")
            sys.exit(1)

    def setup_external_forces(self, mesh, materials, element_materials, Minv, gravity=9.81):
        self.logger.info("Setting up external forces.")
        try:
            g = np.zeros(mesh.dims)
            g[-1] = -gravity
            rho = self.config_manager.get_param("materials", [{}])[0].get("density", 1000.0)
            f, detJeF = pbat.fem.load_vector(mesh, rho * g)
            a = Minv @ f
            return f, a, detJeF
        except Exception as e:
            self.logger.error(f"Failed to set up external forces: {e}")
            sys.exit(1)

    def setup_hyperelastic_potential(self, mesh, materials, element_materials):
        self.logger.info("Setting up hyperelastic potential.")
        try:
            detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
            GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
            Y = np.array([materials[i]["young_modulus"] for i in element_materials])
            nu = np.array([materials[i]["poisson_ratio"] for i in element_materials])
            psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
            hep, _, _, GNeU = pbat.fem.hyper_elastic_potential(mesh, Y=Y, nu=nu, energy=psi)
            return hep, Y, nu, psi, detJeU, GNeU
        except Exception as e:
            self.logger.error(f"Failed to set up hyperelastic potential: {e}")
            sys.exit(1)

    def setup_collision_mesh(self, mesh, V, C, element_materials):
        self.logger.info("Setting up collision mesh.")
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
            self.logger.error(f"Failed to set up collision mesh: {e}")
            sys.exit(1)

    def setup_boundary_conditions(self, mesh, materials, num_nodes_list):
        self.logger.info("Setting up boundary conditions.")
        dirichlet_bc_list = []
        node_offset = 0
        try:
            for _idx, (material, num_nodes) in enumerate(zip(materials, num_nodes_list)):
                percent_fixed = material.get("percent_fixed", 0.0)
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
            self.logger.error(f"Failed to set up boundary conditions: {e}")
            sys.exit(1)

    def setup_collision_potentials(self, dhat, damping_coefficient):
        ipctk.BarrierPotential.use_physical_barrier = True
        barrier_potential = ipctk.BarrierPotential(dhat)
        friction_potential = ipctk.FrictionPotential(damping_coefficient)
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
        communication_client,
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
            "communication_client": (communication_client, ["comm_client"]),
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
            "num_nodes_list": (num_nodes_list, []),
            "running": (False, ["is_running"]),
            "broad_phase_method": (broad_phase_method, ["broad_phase"]),
        }

        # Add attributes to the simulation state
        for key, (value, aliases) in attributes_with_aliases.items():
            simulation_state.add_attribute(key, value, aliases=aliases)
            logger.info(f"Added attribute '{key}' to simulation state.")

        return simulation_state

    def initialize_simulation(self) -> SimulationState:
        """Executes the initialization steps for the simulation.

        Returns
        -------
        SimulationState
            The initialized simulation state.

        """
        try:
            logger = self.logging_manager.logger

            # Extract configuration values using ConfigManager
            inputs = self.config_manager.get_param("inputs", [])
            friction_coefficient = self.config_manager.get_param("friction.friction_coefficient", 0.3)
            damping_coefficient = self.config_manager.get_param("friction.damping_coefficient", 1e-4)
            dhat = self.config_manager.get_param("simulation.dhat", 1e-3)
            dmin = self.config_manager.get_param("simulation.dmin", 1e-4)
            dt = self.config_manager.get_param("simulation.dt", 0.016)
            gravity = self.config_manager.get_param("initial_conditions.gravity", 9.81)
            side_force = self.config_manager.get_param("force.top_force", 10)
            epsv = self.config_manager.get_param("simulation.epsv", 1e-6)

            communication_method = self.config_manager.get_param("communication.method", "redis").lower()
            communication_settings = self.config_manager.get_param("communication.settings.redis", {})

            # Load input meshes and materials
            all_meshes, materials = load_individual_meshes(inputs)
            logger.info(f"Loaded {len(all_meshes)} meshes.")

            # Combine all meshes into a single mesh
            mesh, V, C, element_materials, num_nodes_list = combine_meshes(all_meshes, materials)
            logger.info("Meshes combined into a single mesh.")

            # Setup initial conditions
            x, v, acceleration, n = self.setup_initial_conditions(mesh)

            # Setup mass matrix
            mass_matrix, inverse_mass_matrix = self.setup_mass_matrix(mesh, materials, element_materials)

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
                dhat, damping_coefficient
            )

            # Initialize communication client using NetsFactory (only Redis)
            communication_client = NetsFactory.create_client(
                method=communication_method,
                host=communication_settings.get("host", "localhost"),
                port=communication_settings.get("port", 6379),
                db=communication_settings.get("db", 0),
                password=communication_settings.get("password"),
            )
            logger.info(f"Communication client '{communication_method}' initialized.")

            rho_array = np.array([materials[i]["density"] for i in element_materials])

            # Create connection instances
            network_connection = self.network_factory.create_connection()
            storage_connection = self.storage_factory.create_connection()
            database_connection = self.database_factory.create_connection()

            # Initialize SimulationState
            self.simulation_state = self.initialize_simulation_state(
                self.config,
                mesh,
                x,
                v,
                acceleration,
                mass_matrix,
                hep,
                dt,
                collision_mesh,
                normal_collisions,
                tangential_collisions,
                dhat,
                dmin,
                friction_coefficient,
                epsv,
                damping_coefficient,
                degrees_of_freedom,
                communication_client,
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
            )

            # Add connections to simulation state
            self.simulation_state.set_attribute("network_connection", network_connection)
            self.simulation_state.set_attribute("storage_connection", storage_connection)
            self.simulation_state.set_attribute("database_connection", database_connection)

            logger.info("Simulation state initialized successfully.")

            return self.simulation_state

        except Exception as e:
            logger.error(f"Simulation initialization failed: {e}")
            sys.exit(1)
