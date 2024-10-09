import logging
import math
import sys
import numpy as np
import scipy.sparse as sp
import meshio
import pbatoolkit as pbat
import igl
import ipctk
from redis_interface.redis_client import SimulationRedisClient
from utils.mesh_utils import combine, to_surface, find_codim_vertices
from utils.logging_setup import setup_logging
from materials import Material, add_custom_material
from core.parameters import Parameters
from args import parse_arguments
import json
import collections
import os
import polyscope as ps
import scipy.spatial.transform as spt
from scipy.spatial import cKDTree

# Set environment variables before importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent multiple OpenMP runtimes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

logger = logging.getLogger(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to apply transformations to the vertices
def apply_transformations(vertices, scale, rotation, translation):
    # Convert rotation from quaternion to rotation matrix
    rotation_matrix = spt.Rotation.from_quat(rotation).as_matrix()

    # Apply scaling
    scaling_matrix = np.diag(scale)
    scaled_vertices = vertices @ scaling_matrix

    # Compute the mesh_centroid of the scaled mesh
    mesh_centroid = scaled_vertices.mean(axis=0)

    # Translate vertices to origin for rotation (mesh_centroid-based rotation)
    centered_vertices = scaled_vertices - mesh_centroid

    # Apply rotation
    rotated_vertices = centered_vertices @ rotation_matrix.T

    rotated_vertices += mesh_centroid

    # Apply translation
    translation_vector = np.array(translation).reshape(1, 3)
    transformed_vertices = rotated_vertices + translation_vector

    return transformed_vertices


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
            face_to_element.append(elems[0])
        else:
            logger.warning(f"No element found for face: {face}")
            face_to_element.append(-1)

    return np.array(face_to_element)


def load_json_config(json_path: str, default_dict: dict = None):
    try:
        with open(json_path, 'r') as file:
            config = json.load(file)
        logger.info(f"Configuration loaded successfully from {json_path}.")
        return config
    except FileNotFoundError:
        logger.error(f"JSON configuration file not found: {json_path}")
        return default_dict
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON configuration file: {e}")
        sys.exit(1)


def get_config_value(config, key, default=None):
    try:
        keys = key.split(".")
        value = config
        for k in keys:
            value = value[k]
        return value
    except KeyError:
        return default


def load_and_combine_meshes(inputs):
    """
    Load individual meshes from paths, apply transformations, and combine them into acceleration single mesh.
    """
    all_meshes, materials = load_individual_meshes(inputs)
    combined_mesh, V, C, element_materials = combine_meshes(all_meshes, materials)

    # Extract the number of nodes per mesh for boundary conditions setup
    num_nodes_list = [mesh[0].shape[0] for mesh in all_meshes]

    return combined_mesh, V, C, materials, element_materials, num_nodes_list


# Function to load individual meshes
def load_individual_meshes(inputs):
    all_meshes = []
    materials = []
    for idx, input_entry in enumerate(inputs):
        path = input_entry.get('path')

        # Load material properties
        material_props = input_entry.get('material', {})
        density = material_props.get('density', 1000.0)
        young_modulus = material_props.get('young_modulus', 1e6)
        poisson_ratio = material_props.get('poisson_ratio', 0.45)
        color = material_props.get('color', [255, 255, 255, 1])

        # Load transform properties
        transform = input_entry.get('transform', {})
        scale = transform.get('scale', [1.0, 1.0, 1.0])
        rotation = transform.get('rotation', [0.0, 0.0, 0.0, 1.0])
        translation = transform.get('translation', [0.0, 0.0, 0.0])

        # Load the mesh
        imesh = meshio.read(path)
        vertices = imesh.points.astype(np.float64, order='C')
        cells = imesh.cells_dict.get("tetra")
        if cells is None:
            logger.error(f"No tetrahedral cells found in mesh: {path}")
            continue
        cells = cells.astype(np.int64, order='C')

        # Apply transformations: scale, rotate, and translate
        vertices = apply_transformations(vertices, scale, rotation, translation)

        all_meshes.append((vertices, cells))

        materials.append({
            'density': density,
            'young_modulus': young_modulus,
            'poisson_ratio': poisson_ratio,
            'color': color
        })

        logger.info(f"Mesh {idx + 1} loaded successfully from {path}.")

    return all_meshes, materials


# Function to combine meshes
def combine_meshes(all_meshes, materials):
    V, C = [], []
    element_materials = []

    try:
        for idx, (vertices, cells) in enumerate(all_meshes):
            V.append(vertices)
            C.append(cells)
            num_elements = cells.shape[0]
            element_materials.extend([idx] * num_elements)

        # Combine all vertices and elements into one
        V = np.vstack(V)
        C = np.vstack(C)

        # Deduplicate vertices using KDTree to find close points
        tree = cKDTree(V)
        _, unique_indices = np.unique(tree.query(V, k=1)[1], return_index=True)
        V = V[unique_indices]

        # Update element indices after deduplication
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        C = np.vectorize(vertex_map.get)(C)

        # Create mesh using pbatoolkit
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1
        )
        V, C = mesh.X.T, mesh.E.T
        element_materials = np.array(element_materials, dtype=int)
        logger.info("Meshes combined and deduplicated successfully.")
        return mesh, V, C, element_materials
    except Exception as e:
        logger.error(f"Failed to combine meshes: {e}")
        raise


def setup_initial_conditions(mesh: pbat.fem.Mesh):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
    n = x.shape[0]
    v = np.zeros(n)
    acceleration = np.zeros(n)
    logger.info("Initial conditions set up.")
    return x, v, acceleration, n


def setup_mass_matrix(mesh: pbat.fem.Mesh, materials, element_materials):
    # Create an array of densities per element
    rho_per_element = np.array([materials[mat_idx]['density'] for mat_idx in element_materials])
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    mass_matrix = pbat.fem.MassMatrix(
        mesh, detJeM, rho=rho_per_element[0], dims=3, quadrature_order=2
    ).to_matrix()
    lumpedm = mass_matrix.sum(axis=0)
    mass_matrix = sp.diags(lumpedm, np.array([0]), shape=(mass_matrix.shape[0], mass_matrix.shape[0]))
    inverse_mass_matrix = sp.diags(1.0 / lumpedm, np.array([0]), shape=(mass_matrix.shape[0], mass_matrix.shape[0]))
    logger.info("Mass matrix and its inverse computed.")
    return mass_matrix, inverse_mass_matrix


def setup_external_forces(mesh: pbat.fem.Mesh, materials, inverse_mass_matrix, element_materials, g: float = 9.81, fside: float = 0):
    # Create an array of densities per element
    rho_per_element = np.array([materials[mat_idx]['density'] for mat_idx in element_materials])

    qgf = pbat.fem.inner_product_weights(mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.diags(qgf)
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g_vec = np.zeros(mesh.dims)
    g_vec[-1] = -g  # Gravity in z-direction

    # Add side forces
    side_force_vec = np.zeros(mesh.dims)
    side_force_vec[0] = -fside

    external_force_vector = np.einsum('i,j->ij', rho_per_element, g_vec).flatten(order='F')
    external_force_vector += np.einsum('i,j->ij', rho_per_element, side_force_vec).flatten(order='F')
    f_ext = Nf.T @ Qf @ external_force_vector
    acceleration = inverse_mass_matrix @ f_ext
    logger.info("External forces computed.")
    return acceleration, f_ext, Qf, Nf, qgf


def setup_hyperelastic_potential(mesh: pbat.fem.Mesh, materials, element_materials):
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    young_moduli = np.array([materials[mat_idx]['young_modulus'] for mat_idx in element_materials])
    poisson_ratios = np.array([materials[mat_idx]['poisson_ratio'] for mat_idx in element_materials])
    hyperelastic_energy_model = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hyperelastic_potential = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, young_moduli[0], poisson_ratios[0], energy=hyperelastic_energy_model, quadrature_order=1
    )
    hyperelastic_potential.precompute_hessian_sparsity()
    logger.info("Hyperelastic potential initialized.")
    return hyperelastic_potential, young_moduli, poisson_ratios, hyperelastic_energy_model, detJeU, GNeU


def setup_collision_mesh(mesh: pbat.fem.Mesh, V: np.ndarray, C: np.ndarray, element_materials: np.ndarray):
    boundary_faces = igl.boundary_facets(C)
    edges = ipctk.edges(boundary_faces)
    codim_vertices = find_codim_vertices(mesh, edges)
    if not codim_vertices:
        collision_mesh = ipctk.CollisionMesh.build_from_full_mesh(V, edges, boundary_faces)
    else:
        num_vertices = mesh.X.shape[1]
        is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(num_vertices, edges, codim_vertices)
        collision_mesh = ipctk.CollisionMesh(is_on_surface, edges, boundary_faces)
    logger.info("Collision mesh constructed.")
    collision_constraints = ipctk.Collisions()
    friction_constraints = ipctk.FrictionCollisions()

    face_to_element_mapping = compute_face_to_element_mapping(C, boundary_faces)
    logger.info("Face to element mapping computed.")
    return collision_mesh, edges, boundary_faces, collision_constraints, friction_constraints, face_to_element_mapping


def setup_boundary_conditions(mesh: pbat.fem.Mesh, materials, num_nodes_list):
    dirichlet_bc_list = []
    node_offset = 0
    for idx, (material, num_nodes) in enumerate(zip(materials, num_nodes_list)):
        percent_fixed = material.get('percent_fixed', 0.0)
        # Determine node indices for each mesh
        node_indices = slice(node_offset, node_offset + num_nodes)
        X_sub = mesh.X[:, node_indices]

        Xmin = X_sub.min(axis=1)
        Xmax = X_sub.max(axis=1)
        dX = Xmax - Xmin
        Xmax[-1] = Xmin[-1] + percent_fixed * dX[-1]
        Xmin[-1] = Xmin[-1] - 1e-4
        bounding_box = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
        vertices_in_bounding_box = bounding_box.contained(X_sub)
        dirichlet_bc = np.array(vertices_in_bounding_box)[:, np.newaxis] + node_offset
        dirichlet_bc = np.repeat(dirichlet_bc, mesh.dims, axis=1)
        for d in range(mesh.dims):
            dirichlet_bc[:, d] = mesh.dims * dirichlet_bc[:, d] + d
        dirichlet_bc = dirichlet_bc.reshape(-1)
        dirichlet_bc_list.append(dirichlet_bc)
        node_offset += num_nodes
    # Combine all Dirichlet boundary conditions
    all_dirichlet_bc = np.unique(np.concatenate(dirichlet_bc_list))
    degrees_of_freedom = np.setdiff1d(np.arange(mesh.X.shape[1] * mesh.dims), all_dirichlet_bc)
    logger.info("Boundary conditions applied.")
    return degrees_of_freedom, all_dirichlet_bc


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


def initial_material(name="default", young_modulus=1e6, poisson_ratio=0.45, mass_density=1000.0, color=(0.5, 0.5, 0.5, 1.0)):
    # Check if the material already exists in the enum
    name = name.upper()
    if name in Material.__members__:
        material = Material[name]
    else:
        # Add custom material if it doesn't exist
        add_custom_material(name, young_modulus, poisson_ratio, mass_density, color)
        material = Material[name]  # Retrieve the newly added material from the enum

    logger.info(f"Material '{name}' initialized with properties: Young's Modulus = {young_modulus}, Poisson's Ratio = {poisson_ratio}, Density = {mass_density}, Color = {color}")
    return material


def initialization():
    setup_logging()
    logger = logging.getLogger('SimulationServer')
    args = parse_arguments()

    config = load_json_config(args.json or "", {
        "name": "Default Configuration",
        "inputs": [
            {
                "path": args.input or "meshes/rectangle.mesh",
                "percent_fixed": args.percent_fixed or 0.0,
                "material": {
                    "density": args.mass_density or 1000.0,
                    "young_modulus": args.young_modulus or 1e6,
                    "poisson_ratio": args.poisson_ratio or 0.45,
                    "color": [255, 255, 255, 1]
                },
                "transform": {
                    "scale": [1.0, 1.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "translation": [0.0, 0.0, 0.0]
                }
            }
        ],
        'friction': {
            'friction_coefficient': 0.3,
            'damping_coefficient': 1e-4
        },
        'simulation': {
            'dhat': 1e-3,
            'dmin': 1e-4,
            'dt': 1/60
        },
        'server': {
            'redis_host': args.redis_host or 'localhost',
            'redis_port': args.redis_port or 6379,
            'redis_db': args.redis_db or 0
        },
        "initial_conditions": {
            "gravity": 9.81,
            "side_force": 0,
        }
    })

    inputs = get_config_value(config, 'inputs', [])
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
    mesh, V, C, materials, element_materials, num_nodes_list = load_and_combine_meshes(inputs)

    # Setup initial conditions
    x, v, acceleration, n = setup_initial_conditions(mesh)

    # Mass matrix and external forces (e.g., gravity)
    mass_matrix, inverse_mass_matrix = setup_mass_matrix(mesh, materials, element_materials)

    # Construct load vector from gravity field
    acceleration, f_ext, Qf, Nf, qgf = setup_external_forces(mesh, materials, inverse_mass_matrix, element_materials, g=gravity, fside=side_force)

    # Hyperelastic potential
    hep, Y_array, nu_array, psi, detJeU, GNeU = setup_hyperelastic_potential(mesh, materials, element_materials)

    # IPC contact handling
    cmesh, E, F, cconstraints, fconstraints, face_materials = setup_collision_mesh(mesh, V, C, element_materials)

    # Fix some percentage of the bottom nodes (Dirichlet boundary conditions)
    degrees_of_freedom, dirichlet_boundary_conditions = setup_boundary_conditions(mesh, materials, num_nodes_list)

    # Initialize Redis client
    redis_client = initialize_redis_client(redis_host, redis_port, redis_db)

    material = initial_material()

    ipctk.BarrierPotential.use_physical_barrier = True
    barrier_potential = ipctk.BarrierPotential(dhat)
    friction_potential = ipctk.FrictionPotential(damping_coefficient)

    return (
        config, mesh, x, v, acceleration, mass_matrix, hep, dt, cmesh, cconstraints, fconstraints,
        dhat, dmin, friction_coefficient, damping_coefficient, degrees_of_freedom, redis_client, materials, barrier_potential,
        friction_potential, n, f_ext, Qf, Nf, qgf, Y_array, nu_array, psi,
        detJeU, GNeU, E, F, element_materials, num_nodes_list, face_materials
    )
