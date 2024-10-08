
import logging
import math
import sys
import numpy as np
import scipy as sp
import meshio
import pbatoolkit as pbat
import igl
import ipctk
from redis_interface.redis_client import SimulationRedisClient
from utils.mesh_utils import combine, to_surface, find_codim_vertices
from utils.logging_setup import setup_logging
from materials import Material
from core.parameters import Parameters
from args import parse_arguments
from materials import Material, add_custom_material
import json
import collections


import os

# Set environment variables before importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent multiple OpenMP runtimes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


logger = logging.getLogger(__name__)

# ipctk.set_num_threads(10)

def compute_face_to_element_mapping(C: np.ndarray, F: np.ndarray) -> np.ndarray:
    face_dict = collections.defaultdict(list)
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
            face_dict[face].append(elem_idx)

    # Map each boundary face to the corresponding element
    face_to_element = []
    for face in F:
        face_key = tuple(sorted(face))
        elems = face_dict.get(face_key, [])
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
    V, C = [], []
    materials = []
    element_materials = []
    num_nodes_list = []
    try:
        total_meshes = len(inputs)
        vertex_offset = 0
        for idx, input_entry in enumerate(inputs):
            path = input_entry.get('path')
            percent_fixed = input_entry.get('percent_fixed', 0.0)

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

            imesh = meshio.read(path)
            V1 = imesh.points.astype(np.float64, order='C')
            C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')

            # Apply scaling, rotation, and translation to the mesh
            # R = sp.spatial.transform.Rotation.from_quat(rotation).as_matrix()
            # V1 = V1 @ np.diag(scale) @ R.T + np.array(translation)

            V.append(V1)
            C.append(C1)

            for c in range(1):
                R = sp.spatial.transform.Rotation.from_quat(
                    [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
                V2 = (V[-1] - V[-1].mean(axis=0)) @ R.T + V[-1].mean(axis=0)
                V2[:, 2] += (V2[:, 2].max() - V2[:, 2].min()) + 50
                C2 = C[-1]
                V.append(V2)
                C.append(C2)
            
            materials.append({
                'density': density,
                'young_modulus': young_modulus,
                'poisson_ratio': poisson_ratio,
                'color': color,
                'percent_fixed': percent_fixed
            })
            
            input_entry['num_nodes'] = V1.shape[0]
            num_nodes_list.append(V1.shape[0])
            logger.info(f"Mesh {idx + 1}/{total_meshes} loaded successfully from {path}.")

            # Keep track of which elements correspond to this material
            num_elements = C1.shape[0]
            element_materials.extend([idx] * num_elements)


        V, C = combine(V, C)
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
        V, C = mesh.X.T, mesh.E.T
        element_materials = np.array(element_materials, dtype=int)
        logger.info("Meshes loaded and combined successfully.")
        return mesh, V, C, materials, element_materials, num_nodes_list
    except Exception as e:
        logger.error(f"Failed to load and combine meshes: {e}")
        sys.exit(1)

def setup_initial_conditions(mesh: pbat.fem.Mesh):
    x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
    n = x.shape[0]
    v = np.zeros(n)
    a = np.zeros(n)
    logger.info("Initial conditions set up.")
    return x, v, a, n 

def setup_mass_matrix(mesh: pbat.fem.Mesh, materials, element_materials):
    # Create an array of densities per element
    rho_per_element = np.array([materials[mat_idx]['density'] for mat_idx in element_materials])
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    M = pbat.fem.MassMatrix(
        mesh, detJeM, rho=rho_per_element[0], dims=3, quadrature_order=2
    ).to_matrix()
    lumpedm = M.sum(axis=0)
    M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    Minv = sp.sparse.spdiags(1.0 / lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    logger.info("Mass matrix and its inverse computed.")
    return M, Minv

def setup_external_forces(mesh: pbat.fem.Mesh, materials, Minv, element_materials, g: float = 9.81, fside: float = 0):
    # Create an array of densities per element
    rho_per_element = np.array([materials[mat_idx]['density'] for mat_idx in element_materials])

    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags(qgf)
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g_vec = np.zeros(mesh.dims)
    g_vec[-1] = -g  # Gravity in z-direction

    # Add side forces
    side_force_vec = np.zeros(mesh.dims)
    side_force_vec[0] = -fside

    fe = np.tile(rho_per_element[0] * g_vec[:, np.newaxis], mesh.E.shape[1])
    fe += np.tile(rho_per_element[0] * side_force_vec[:, np.newaxis], mesh.E.shape[1])
    f_ext = fe @ Qf @ Nf
    f_ext = f_ext.reshape(-1, order="F")
    a = Minv @ f_ext
    logger.info("External forces computed.")
    return a, f_ext, Qf, Nf, qgf

def setup_hyperelastic_potential(mesh: pbat.fem.Mesh, materials, element_materials):
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y_array = np.array([materials[mat_idx]['young_modulus'] for mat_idx in element_materials])
    nu_array = np.array([materials[mat_idx]['poisson_ratio'] for mat_idx in element_materials])
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y_array[0], nu_array[0], energy=psi, quadrature_order=1
    )
    hep.precompute_hessian_sparsity()
    logger.info("Hyperelastic potential initialized.")
    return hep, Y_array, nu_array, psi, detJeU, GNeU

def setup_collision_mesh(mesh: pbat.fem.Mesh, V: np.ndarray, C: np.ndarray, element_materials: np.ndarray):
    F = igl.boundary_facets(C)
    E = ipctk.edges(F)
    codim_vertices = find_codim_vertices(mesh, E)
    if not codim_vertices:
        cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    else:
        n_vertices = mesh.X.shape[1]
        is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(n_vertices, E, codim_vertices)
        cmesh = ipctk.CollisionMesh(is_on_surface, E, F)
    logger.info("Collision mesh constructed.")
    cconstraints = ipctk.Collisions()
    fconstraints = ipctk.FrictionCollisions()

    face_materials = compute_face_to_element_mapping(C, F)
    logger.info("Face to element mapping computed.")
    return cmesh, E, F, cconstraints, fconstraints, face_materials


def setup_boundary_conditions(mesh: pbat.fem.Mesh, materials, num_nodes_list):
    dbcs_list = []
    node_offset = 0
    for idx, (mat, num_nodes) in enumerate(zip(materials, num_nodes_list)):
        percent_fixed = mat['percent_fixed']
        # Determine node indices for each mesh
        node_indices = slice(node_offset, node_offset + num_nodes)
        X_sub = mesh.X[:, node_indices]

        Xmin = X_sub.min(axis=1)
        Xmax = X_sub.max(axis=1)
        dX = Xmax - Xmin
        Xmax[-1] = Xmin[-1] + percent_fixed * dX[-1]
        Xmin[-1] = Xmin[-1] - 1e-4
        aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
        vdbc = aabb.contained(X_sub)
        dbcs = np.array(vdbc)[:, np.newaxis] + node_offset
        dbcs = np.repeat(dbcs, mesh.dims, axis=1)
        for d in range(mesh.dims):
            dbcs[:, d] = mesh.dims * dbcs[:, d] + d
        dbcs = dbcs.reshape(-1)
        dbcs_list.append(dbcs)
        node_offset += num_nodes
    # Combine all Dirichlet boundary conditions
    all_dbcs = np.unique(np.concatenate(dbcs_list))
    dofs = np.setdiff1d(np.arange(mesh.X.shape[1] * mesh.dims), all_dbcs)
    logger.info("Boundary conditions applied.")
    return dofs, all_dbcs

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
    from utils.logging_setup import setup_logging
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
                    "rotation": [0.0, 0.0, 0.0, 0.0],
                    "translation": [0.0, 0.0, 0.0]
                }
            }
        ],
        'friction': {
            'mu': 0.3,
            'epsv': 1e-4
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

    mu = get_config_value(config, 'friction.coefficient', 0.3)
    epsv = get_config_value(config, 'friction.damping', 1e-4)
    dhat = get_config_value(config, 'simulation.collision_threshold', 1e-3)
    dmin = get_config_value(config, 'simulation.min_distance', 1e-4)
    dt = get_config_value(config, 'simulation.time_step', 0.016)
    redis_host = get_config_value(config, 'server.redis_host', 'localhost')
    redis_port = get_config_value(config, 'server.redis_port', 6379)
    redis_db = get_config_value(config, 'server.redis_db', 0)
    gravity = get_config_value(config, 'initial_conditions.gravity', 9.81)
    side_force = get_config_value(config, 'initial_conditions.side_force', 10)


    # Load input meshes and combine them into 1 mesh
    mesh, V, C, materials, element_materials, num_nodes_list = load_and_combine_meshes(inputs)

    # Setup initial conditions
    x, v, a, n = setup_initial_conditions(mesh)

    # Mass matrix and external forces (e.g., gravity)
    M, Minv = setup_mass_matrix(mesh, materials, element_materials)

    # Construct load vector from gravity field
    a, f_ext, Qf, Nf, qgf = setup_external_forces(mesh, materials, Minv, element_materials, g=gravity, fside=side_force)

    # Hyperelastic potential
    hep, Y_array, nu_array, psi, detJeU, GNeU = setup_hyperelastic_potential(mesh, materials, element_materials)

    # IPC contact handling
    cmesh, E, F, cconstraints, fconstraints, face_materials = setup_collision_mesh(mesh, V, C, element_materials)

    # Fix some percentage of the bottom nodes (Dirichlet boundary conditions)
    dofs, dbcs = setup_boundary_conditions(mesh, materials, num_nodes_list)

    # Initialize Redis client
    redis_client = initialize_redis_client(redis_host, redis_port, redis_db)

    material = initial_material()

    ipctk.BarrierPotential.use_physical_barrier = True
    barrier_potential = ipctk.BarrierPotential(dhat)
    friction_potential = ipctk.FrictionPotential(epsv)

    return (
        config, mesh, x, v, a, M, hep, dt, cmesh, cconstraints, fconstraints,
        dhat, dmin, mu, epsv, dofs, redis_client, materials, barrier_potential,
        friction_potential, n, f_ext, Qf, Nf, qgf, Y_array, nu_array, psi,
        detJeU, GNeU, E, F, element_materials, num_nodes_list, face_materials
    )