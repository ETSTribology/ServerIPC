
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
from materials import Material

logger = logging.getLogger(__name__)

ipctk.set_num_threads(10)

def load_and_combine_meshes(input_path: str, num_copies: int):
    V, C = [], []
    try:
        imesh = meshio.read(input_path)
        V1 = imesh.points.astype(np.float64, order='C')
        C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')
        V.append(V1)
        C.append(C1)
        for c in range(num_copies):
            V2 = (V[-1] - V[-1].mean(axis=0)) + V[-1].mean(axis=0)
            V2[:, 2] += 5
            V2[:, 0] += 5
            C2 = C[-1]
            V.append(V2)
            C.append(C2)

        V, C = combine(V, C)
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
        V, C = mesh.X.T, mesh.E.T
        logger.info("Mesh loaded and combined successfully.")
        return mesh, V, C
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

def setup_mass_matrix(mesh: pbat.fem.Mesh, rho: float):
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    M = pbat.fem.MassMatrix(
        mesh, detJeM, rho=rho, dims=3, quadrature_order=2
    ).to_matrix()
    lumpedm = M.sum(axis=0)
    M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    Minv = sp.sparse.spdiags(1.0 / lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    logger.info("Mass matrix and its inverse computed.")
    return M, Minv

def setup_external_forces(mesh: pbat.fem.Mesh, rho: float, Minv: sp.sparse.dia_matrix):
    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags(qgf)
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g_vec = np.zeros(mesh.dims)
    g_vec[-1] = -9.81  # Gravity in z-direction

    # Add side forces
    side_force = -10
    side_force_vec = np.zeros(mesh.dims)
    side_force_vec[0] = side_force

    fe = np.tile(rho * g_vec[:, np.newaxis], mesh.E.shape[1])
    fe += np.tile(rho * side_force_vec[:, np.newaxis], mesh.E.shape[1])
    f_ext = fe @ Qf @ Nf
    f_ext = f_ext.reshape(math.prod(f_ext.shape), order="F")
    a = Minv @ f_ext
    logger.info("External forces computed.")
    return a, f_ext, Qf, Nf, qgf

def setup_hyperelastic_potential(mesh: pbat.fem.Mesh, Y: float, nu: float):
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.full(mesh.E.shape[1], Y)
    nu = np.full(mesh.E.shape[1], nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=1
    )
    hep.precompute_hessian_sparsity()
    logger.info("Hyperelastic potential initialized.")
    return hep, Y, nu, psi, detJeU, GNeU

def setup_collision_mesh(mesh: pbat.fem.Mesh, V: np.ndarray, C: np.ndarray):
    F = igl.boundary_facets(C)
    E = ipctk.edges(F)
    codim_vertices = find_codim_vertices(mesh, E)
    if not codim_vertices:
        logger.warning("No codimensional vertices found. All vertices are on the surface.")
        cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    else:
        n_vertices = mesh.X.shape[1]
        is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(n_vertices, E, codim_vertices)
        cmesh = ipctk.CollisionMesh(is_on_surface, E, F)
    logger.info("Collision mesh constructed.")
    cconstraints = ipctk.Collisions()
    fconstraints = ipctk.FrictionCollisions()
    return cmesh, E, F, cconstraints, fconstraints

def setup_boundary_conditions(mesh: pbat.fem.Mesh, percent_fixed: float):
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    dX = Xmax - Xmin
    Xmax[-1] = Xmin[-1] + percent_fixed * dX[-1]
    Xmin[-1] = Xmin[-1] - 1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims * dbcs[:, d] + d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    dofs = np.setdiff1d(list(range(mesh.X.shape[1] * mesh.dims)), dbcs)
    logger.info("Boundary conditions applied.")
    return dofs, dbcs, Xmin, Xmax, aabb, vdbc, dX

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

def initial_material():
    return Material.DEFAULT

def initialization():
    from utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger('SimulationServer')

    args = parse_arguments()

    dhat = 1e-3
    dmin = 1e-4
    mu = 0.3
    epsv = 1e-4
    dt = 1/60

    # Load input meshes and combine them into 1 mesh
    mesh, V, C = load_and_combine_meshes(args.input, args.copy)

    # Setup initial conditions
    x, v, a, n = setup_initial_conditions(mesh)

    # Mass matrix and external forces (e.g., gravity)
    rho = args.mass_density
    M, Minv = setup_mass_matrix(mesh, args.mass_density)

    # Construct load vector from gravity field
    a, f_ext, Qf, Nf, qgf = setup_external_forces(mesh, rho, Minv)

    # Hyperelastic potential
    hep, Y, nu, psi, detJeU, GNeU = setup_hyperelastic_potential(mesh, args.young_modulus, args.poisson_ratio)

    # IPC contact handling
    cmesh, E, F, cconstraints, fconstraints = setup_collision_mesh(mesh, V, C)

    # Fix some percentage of the bottom nodes (Dirichlet boundary conditions)
    dofs, dbcs, Xmin, Xmax, aabb, vdbc, dX = setup_boundary_conditions(mesh, args.percent_fixed)

    # Initialize Redis client
    redis_client = initialize_redis_client(args.redis_host, args.redis_port, args.redis_db)

    material = initial_material()

    ipctk.BarrierPotential.use_physical_barrier = True
    barrier_potential = ipctk.BarrierPotential(dhat)
    friction_potential = ipctk.FrictionPotential(epsv)

    return args, mesh, x, v, a, M, hep, dt, cmesh, cconstraints, fconstraints, dhat, dmin, mu, epsv, dofs, redis_client, material, barrier_potential, friction_potential, n, f_ext, Qf, Nf, qgf, Y, nu, psi, detJeU, GNeU, E, F, aabb, vdbc, dX