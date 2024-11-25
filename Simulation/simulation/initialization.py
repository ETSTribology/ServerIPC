import logging
import math
import sys
import numpy as np
import scipy as sp
import meshio
import pbatoolkit as pbat
import igl
import ipctk
from net_interface.redis_client import SimulationRedisClient
from utils.mesh_utils import combine, to_surface, find_codim_vertices
from utils.logging_setup import setup_logging
from materials import Material
from args import parse_arguments
import collections
import os
import scipy.spatial.transform as spt
from utils.config import generate_default_config, load_config, get_config_value
from utils.load import load_individual_meshes, combine_meshes, load_individual_meshes_with_instancing
from materials.materials import materials, add_custom_material

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

def compute_face_to_element_mapping(C: np.ndarray, F: np.ndarray) -> np.ndarray:
    face_to_element_dict = collections.defaultdict(list)
    # Iterate over each tetrahedron and its faces
    for elem_idx, tet in enumerate(C):
        # Define all four faces of the tetrahedron
        tet_faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in tet_faces:
            face_to_element_dict[face].append(elem_idx)

    # Map each boundary face to the corresponding element
    face_to_element = []
    for face in F:
        face_key = tuple(sorted(face))
        elems = face_to_element_dict.get(face_key, [])
        if len(elems) == 1:
            face_to_element.append(elems[0])
        elif len(elems) > 1:
            face_to_element.append(elems[0])  # You might want to handle shared faces differently
        else:
            logger.warning(f"No element found for face: {face}")
            face_to_element.append(-1)

    return np.array(face_to_element)

def setup_initial_conditions(mesh: pbat.fem.Mesh):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
    n = x.shape[0]
    v = np.zeros(n)
    acceleration = np.zeros(n)
    logger.info("Initial conditions set up.")
    return x, v, acceleration, n


def setup_mass_matrix(mesh, materials, element_materials):
    # Construct the mass matrix with lumping
    M, detJe = pbat.fem.mass_matrix(
        mesh,
        rho=1000,
        lump=True
    )
    # Inverse of lumped mass matrix (diagonal)
    Minv = sp.sparse.diags(1./M.diagonal())
    return M, Minv



def setup_external_forces(mesh, materials, element_materials, Minv, gravity=9.81):
    g = np.zeros(mesh.dims)
    g[-1] = -gravity
    rho = 1000
    f, detJeF = pbat.fem.load_vector(mesh, rho*g)
    a = Minv @ f
    
    return f, a, detJeF

def setup_hyperelastic_potential(mesh, materials, element_materials):
    logger.info("Hyperelastic potential setup started.")
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.array([materials[i]['young_modulus'] for i in element_materials])
    nu = np.array([materials[i]['poisson_ratio'] for i in element_materials])
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep, egU, wgU, GNeU = pbat.fem.hyper_elastic_potential(
        mesh, Y=Y, nu=nu, energy=psi)
    logger.info("Hyperelastic potential setup completed.")
    return hep, Y, nu, psi, detJeU, GNeU


def setup_collision_mesh(mesh: pbat.fem.Mesh, V: np.ndarray, C: np.ndarray, element_materials: np.ndarray, debug=False):
    boundary_faces = igl.boundary_facets(C)
    edges = ipctk.edges(boundary_faces)
    codim_vertices = find_codim_vertices(mesh, edges)

    if not codim_vertices:
        logger.warning("No codimensional vertices found. This may be normal depending on your mesh structure.")
        collision_mesh = ipctk.CollisionMesh.build_from_full_mesh(V, edges, boundary_faces)
    else:
        logger.info("Constructing collision mesh...")
        num_vertices = mesh.X.shape[1]
        is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(num_vertices, edges, codim_vertices)
        collision_mesh = ipctk.CollisionMesh(is_on_surface, edges, boundary_faces)

    logger.info("Collision mesh constructed.")
    collision_constraints = ipctk.NormalCollisions()
    friction_constraints = ipctk.TangentialCollisions()

    face_to_element_mapping = compute_face_to_element_mapping(C, boundary_faces)
    logger.info("Face to element mapping computed.")

    return collision_mesh, edges, boundary_faces, collision_constraints, friction_constraints, face_to_element_mapping

def setup_boundary_conditions(mesh: pbat.fem.Mesh, materials, num_nodes_list):
    dirichlet_bc_list = []  # To collect Dirichlet boundary condition indices for all parts
    node_offset = 0  # Keeps track of where the nodes start for each part of the mesh

    # Iterate over materials and nodes
    for idx, (material, num_nodes) in enumerate(zip(materials, num_nodes_list)):
        percent_fixed = material.get('percent_fixed', 0.0)  # Get the percent_fixed property (default 0.0)

        if percent_fixed > 0.0:
            # Determine node indices for this part of the mesh
            node_indices = slice(node_offset, node_offset + num_nodes)
            X_sub = mesh.X[:, node_indices]  # Subset of node positions for this part

            # Find the min and max coordinates in the Z direction (for vertical fixing)
            Xmin = X_sub.min(axis=1)
            Xmax = X_sub.max(axis=1)
            dX = Xmax - Xmin

            # Compute the bounding box in the Z-direction based on the percent_fixed
            Xmax[-1] = Xmin[-1] + percent_fixed * dX[-1]
            Xmin[-1] = Xmin[-1] - 1e-4  # Small buffer for numerical stability

            # Find the nodes within this bounding box
            bounding_box = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
            vertices_in_bounding_box = bounding_box.contained(X_sub)

            if len(vertices_in_bounding_box) > 0:
                # Convert the vertex indices into DOF indices for Dirichlet boundary conditions
                dirichlet_bc = np.array(vertices_in_bounding_box)[:, np.newaxis] + node_offset
                dirichlet_bc = np.repeat(dirichlet_bc, mesh.dims, axis=1)

                # Adjust DOF indices for the number of dimensions
                for d in range(mesh.dims):
                    dirichlet_bc[:, d] = mesh.dims * dirichlet_bc[:, d] + d

                # Reshape and add the result to the boundary condition list
                dirichlet_bc = dirichlet_bc.reshape(-1)
                dirichlet_bc_list.append(dirichlet_bc)

        # Update node offset for the next part of the mesh
        node_offset += num_nodes

    # Combine all Dirichlet boundary conditions and ensure uniqueness
    all_dirichlet_bc = np.unique(np.concatenate(dirichlet_bc_list)) if dirichlet_bc_list else np.array([])

    # Calculate total degrees of freedom excluding fixed nodes
    total_dofs = np.setdiff1d(np.arange(mesh.X.shape[1] * mesh.dims), all_dirichlet_bc)

    logger.info(f"Boundary conditions applied for {len(dirichlet_bc_list)} regions.")

    return total_dofs, all_dirichlet_bc


def initialize_redis_client(host: str, port: int, db: int):
    try:
        redis_client = SimulationRedisClient(
            host=host,
            port=port,
            db=db
        )
        logger.info("Redis client initialized.")
        return redis_client
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
        sys.exit(1)


def initial_material(name="default", young_modulus=1e6, poisson_ratio=0.45, density=1000.0, color=(128, 128, 128, 1)):
    name = name.upper()
    if name in materials:
        material = materials[name]
    else:
        # Add custom material if it doesn't exist
        add_custom_material(name, young_modulus, poisson_ratio, density, color)
        material = materials[name]
    logger.info(f"Material '{name}' initialized with properties: Young's Modulus = {material.young_modulus}, Poisson's Ratio = {material.poisson_ratio}, Density = {material.density}, Color = {material.color}")
    return material


def initialization():
    setup_logging()
    logger = logging.getLogger('SimulationServer')
    args = parse_arguments()

    config = load_config(args.json or "", generate_default_config(args))

    logger.info(f"Loaded configuration: {config}")

    inputs = get_config_value(config, 'inputs', [])

    logger.info("Initializing simulation...")

    if not isinstance(inputs, list):
        inputs = [inputs]

    friction_coefficient = get_config_value(config, 'friction.friction_coefficient', 0.3)
    damping_coefficient = get_config_value(config, 'friction.damping_coefficient', 1e-4)
    dhat = get_config_value(config, 'simulation.dhat', 1e-3)
    dmin = get_config_value(config, 'simulation.dmin', 1e-4)
    dt = get_config_value(config, 'simulation.dt', 0.016)
    redis_host = get_config_value(config, 'server.redis_host', 'localhost')
    redis_port = get_config_value(config, 'server.redis_port', 6379)
    redis_db = get_config_value(config, 'server.redis_db', 0)
    gravity = get_config_value(config, 'initial_conditions.gravity', 9.81)
    side_force = get_config_value(config, 'initial_conditions.side_force', 10)

    # Load input meshes and combine them into 1 mesh
    all_meshes, materials = load_individual_meshes(inputs)

    # Combine all meshes into a single mesh
    mesh, V, C, element_materials, num_nodes_list, instances = combine_meshes(all_meshes, materials, False)

    # Setup initial conditions
    x, v, acceleration, n = setup_initial_conditions(mesh)

    # Mass matrix
    mass_matrix, inverse_mass_matrix = setup_mass_matrix(mesh, materials, element_materials)

    # Compute the total external forces (gravity only)
    f_ext, acceleration, qgf = setup_external_forces(mesh, materials, element_materials, inverse_mass_matrix, gravity)

    # Hyperelastic potential
    hep, Y_array, nu_array, psi, detJeU, GNeU = setup_hyperelastic_potential(mesh, materials, element_materials)

    # IPC contact handling
    cmesh, E, F, cconstraints, fconstraints, face_materials = setup_collision_mesh(mesh, V, C, element_materials)

    # Fix some percentage of the bottom nodes (Dirichlet boundary conditions)
    degrees_of_freedom, dirichlet_boundary_conditions = setup_boundary_conditions(mesh, materials, num_nodes_list)

    # Initialize Redis client
    redis_client = initialize_redis_client(redis_host, redis_port, redis_db)

    ipctk.BarrierPotential.use_physical_barrier = True
    barrier_potential = ipctk.BarrierPotential(dhat)
    friction_potential = ipctk.FrictionPotential(damping_coefficient)

    return (
        config, mesh, x, v, acceleration, mass_matrix, hep, dt, cmesh, cconstraints, fconstraints,
        dhat, dmin, friction_coefficient, damping_coefficient, degrees_of_freedom, redis_client, materials, barrier_potential,
        friction_potential, n, f_ext, qgf, Y_array, nu_array, psi,
        detJeU, GNeU, E, F, element_materials, num_nodes_list, face_materials, instances
    )