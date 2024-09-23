import argparse
import logging
import math
import numpy as np
import scipy as sp
import meshio
import pbatoolkit as pbat
import igl
import ipctk
from redis_interface.redis_client import SimulationRedisClient
from loop import run_simulation
from utils.mesh_utils import combine, to_surface, find_codim_vertices
from utils.logging_setup import setup_logging
import sys

logger = logging.getLogger(__name__)

ipctk.set_num_threads(10)

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="3D Elastic Simulation of Linear FEM Tetrahedra using IPC",
        description="Simulate 3D elastic deformations with contact handling."
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", required=True)
    parser.add_argument("--percent-fixed", type=float, default=0.1,
                        help="Percentage of input mesh's bottom to fix")
    parser.add_argument("-m", "--mass-density", type=float, default=1000.0,
                        help="Mass density")
    parser.add_argument("-Y", "--young-modulus", type=float, default=6e9,
                        help="Young's modulus")
    parser.add_argument("-n", "--poisson-ratio", type=float, default=0.45,
                        help="Poisson's ratio")
    parser.add_argument("-c", "--copy", type=int, default=1,
                        help="Number of copies of input model")
    parser.add_argument("--redis-host", type=str, default="localhost",
                        help="Redis host address")
    parser.add_argument("--redis-port", type=int, default=6379,
                        help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0,
                        help="Redis database")
    return parser.parse_args()

def main():
    # Setup logging
    from utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger('SimulationServer')

    args = parse_arguments()

    # Load input meshes and combine them into 1 mesh
    V, C = [], []
    try:
        imesh = meshio.read(args.input)
        V1 = imesh.points.astype(np.float64, order='C')
        C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')
        V.append(V1)
        C.append(C1)
        for c in range(args.copy):
            V2 = (V[-1] - V[-1].mean(axis=0)) + V[-1].mean(axis=0)
            V2[:, 2] += 3
            V2[:, 0] += 5
            C2 = C[-1]
            V.append(V2)
            C.append(C2)

        V, C = combine(V, C)
        mesh = pbat.fem.Mesh(
            V.T, C.T, element=pbat.fem.Element.Tetrahedron, order=1)
        V, C = mesh.X.T, mesh.E.T

        # Setup initial conditions
        x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
        n = x.shape[0]
        v = np.zeros(n)
        logger.info("Mesh loaded and combined successfully.")
    except Exception as e:
        logger.error(f"Failed to load and combine meshes: {e}")
        sys.exit(1)

    # Setup initial conditions
    x = mesh.X.reshape(math.prod(mesh.X.shape), order="F")
    n = x.shape[0]
    v = np.zeros(n)

    # Mass matrix and external forces (e.g., gravity)
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.mass_density
    M = pbat.fem.MassMatrix(
        mesh, detJeM, rho=rho, dims=3, quadrature_order=2
    ).to_matrix()
    lumpedm = M.sum(axis=0)
    M = sp.sparse.spdiags(lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])
    Minv = sp.sparse.spdiags(1.0 / lumpedm, np.array([0]), m=M.shape[0], n=M.shape[0])

    # Construct load vector from gravity field
    qgf = pbat.fem.inner_product_weights(
        mesh, quadrature_order=1).flatten(order="F")
    Qf = sp.sparse.diags(qgf)
    Nf = pbat.fem.shape_function_matrix(mesh, quadrature_order=1)
    g_vec = np.zeros(mesh.dims)
    g_vec[-1] = -9.81  # Gravity in z-direction

    # Add side forces
    side_force = -9.81
    side_force_vec = np.zeros(mesh.dims)
    side_force_vec[0] = side_force

    fe = np.tile(rho * g_vec[:, np.newaxis], mesh.E.shape[1])
    fe += np.tile(rho * side_force_vec[:, np.newaxis], mesh.E.shape[1])
    f_ext = fe @ Qf @ Nf
    f_ext = f_ext.reshape(math.prod(f_ext.shape), order="F")
    a = Minv @ f_ext

    # Hyperelastic potential
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.full(mesh.E.shape[1], args.young_modulus)
    nu = np.full(mesh.E.shape[1], args.poisson_ratio)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=1
    )
    hep.precompute_hessian_sparsity()
    logger.info("Hyperelastic potential initialized.")

    # IPC contact handling
    F = igl.boundary_facets(C)
    E = ipctk.edges(F)
    codim_vertices = find_codim_vertices(mesh, E)
    if not codim_vertices:
        logger.warning("No codimensional vertices found. All vertices are on the surface.")
        cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    else:
        n_vertices = mesh.X.shape[1]
        is_on_surface = ipctk.CollisionMesh.construct_is_on_surface(n_vertices, E, codim_vertices)
        cmesh = ipctk.CollisionMesh(is_on_surface, x, E, F)
    cconstraints = ipctk.Collisions()
    fconstraints = ipctk.FrictionCollisions()

    dhat = 1e-3
    dmin = 1e-4
    mu = 0.3
    epsv = 1e-4

    # Fix some percentage of the bottom nodes (Dirichlet boundary conditions)
    Xmin = mesh.X.min(axis=1)
    Xmax = mesh.X.max(axis=1)
    dX = Xmax - Xmin
    Xmax[-1] = Xmin[-1] + args.percent_fixed * dX[-1]
    Xmin[-1] = Xmin[-1] - 1e-4
    aabb = pbat.geometry.aabb(np.vstack((Xmin, Xmax)).T)
    vdbc = aabb.contained(mesh.X)
    dbcs = np.array(vdbc)[:, np.newaxis]
    dbcs = np.repeat(dbcs, mesh.dims, axis=1)
    for d in range(mesh.dims):
        dbcs[:, d] = mesh.dims * dbcs[:, d] + d
    dbcs = dbcs.reshape(math.prod(dbcs.shape))
    dofs = np.setdiff1d(list(range(n)), dbcs)

    logger.info("Boundary conditions applied.")

    # Initialize Redis client
    try:
        redis_client = SimulationRedisClient(
            host=args.redis_host,
            port=args.redis_port,
            db=args.redis_db
        )
    except Exception as e:
        logger.error(f"Failed to initialize Redis client: {e}")
        sys.exit(1)

    from materials import Material
    material = Material.DEFAULT

    # Run the simulation
    run_simulation(
        mesh=mesh,
        x=x,
        v=v,
        a=a,
        M=M,
        hep=hep,
        dt=0.01,
        cmesh=cmesh,
        cconstraints=cconstraints,
        fconstraints=fconstraints,
        dhat=dhat,
        dmin=dmin,
        mu=mu,
        epsv=epsv,
        dofs=dofs,
        redis_client=redis_client,
        material=material
    )

if __name__ == "__main__":
    main()
