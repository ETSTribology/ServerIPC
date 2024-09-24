import redis
import pickle
import numpy as np
import scipy as sp
import meshio
import pbatoolkit as pbat
import igl
import ipctk
import argparse
import math
import itertools
import time
import zlib
import bson
import base64
import threading
from collections.abc import Callable
import logging
import sys

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


logger = logging.getLogger(__name__)

# Helper functions and classes (from your original code)
def combine(V: list, C: list):
    Vsizes = [Vi.shape[0] for Vi in V]
    offsets = [0] + list(itertools.accumulate(Vsizes))
    C = [C[i] + offsets[i] for i in range(len(C))]
    C = np.vstack(C)
    V = np.vstack(V)
    return V, C

def line_search(alpha0: float,
                xk: np.ndarray,
                dx: np.ndarray,
                gk: np.ndarray,
                f: Callable[[np.ndarray], float],
                maxiters: int = 20,
                c: float = 1e-4,
                tau: float = 0.5):
    alphaj = alpha0
    Dfk = gk.dot(dx)
    fk = f(xk)
    for j in range(maxiters):
        fx = f(xk + alphaj*dx)
        flinear = fk + alphaj * c * Dfk
        if fx <= flinear:
            break
        alphaj = tau*alphaj
    return alphaj

def newton(x0: np.ndarray,
           f: Callable[[np.ndarray], float],
           grad: Callable[[np.ndarray], np.ndarray],
           hess: Callable[[np.ndarray], sp.sparse.csc_matrix],
           lsolver: Callable[[sp.sparse.csc_matrix, np.ndarray], np.ndarray],
           alpha0: Callable[[np.ndarray, np.ndarray], float],
           maxiters: int = 10,
           rtol: float = 1e-5,
           callback: Callable[[np.ndarray], None] = None):
    xk = x0.copy()
    gk = grad(xk)
    for k in range(maxiters):
        gnorm = np.linalg.norm(gk, 1)
        if gnorm < rtol:
            break
        Hk = hess(xk)
        dx = lsolver(Hk, -gk)
        alpha = line_search(alpha0(xk, dx), xk, dx, gk, f)
        xk = xk + alpha*dx
        gk = grad(xk)
        if callback is not None:
            callback(xk)
    return xk

def to_surface(x: np.ndarray, mesh: pbat.fem.Mesh, cmesh: ipctk.CollisionMesh):
    X = x.reshape(mesh.X.shape[0], mesh.X.shape[1], order="F").T
    XB = cmesh.map_displacements(X)
    return XB

class Parameters():
    def __init__(self,
                 mesh: pbat.fem.Mesh,
                 xt: np.ndarray,
                 vt: np.ndarray,
                 a: np.ndarray,
                 M: sp.sparse.dia_array,
                 hep: pbat.fem.HyperElasticPotential,
                 dt: float,
                 cmesh: ipctk.CollisionMesh,
                 cconstraints: ipctk.CollisionConstraints,
                 fconstraints: ipctk.FrictionConstraints,
                 dhat: float = 1e-3,
                 dmin: float = 1e-4,
                 mu: float = 0.3,
                 epsv: float = 1e-4):
        self.mesh = mesh
        self.xt = xt
        self.vt = vt
        self.a = a
        self.M = M
        self.hep = hep
        self.dt = dt
        self.cmesh = cmesh
        self.cconstraints = cconstraints
        self.fconstraints = fconstraints
        self.dhat = dhat
        self.dmin = dmin
        self.mu = mu
        self.epsv = epsv

        self.dt2 = dt**2
        self.xtilde = xt + dt*vt + self.dt2 * a
        self.avgmass = M.diagonal().mean()
        self.kB = None
        self.maxkB = None
        self.dprev = None
        self.dcurrent = None
        BX = to_surface(xt, mesh, cmesh)
        self.bboxdiag = ipctk.world_bbox_diagonal_length(BX)
        self.gU = None
        self.gB = None

class Potential():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> float:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=False, hessian=False)
        U = hep.eval()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        fconstraints.build(cmesh, BX, cconstraints, dhat, kB, mu)
        EB = cconstraints.compute_potential(cmesh, BX, dhat)
        EF = fconstraints.compute_potential(cmesh, BXdot, epsv)
        return 0.5 * (x - xtilde).T @ M @ (x - xtilde) + dt2*U + kB * EB + dt2*EF

class Gradient():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        xtilde = self.params.xtilde
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=True, hessian=False)
        gU = hep.gradient()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        cconstraints.build(cmesh, BX, dhat, dmin=dmin)
        gB = cconstraints.compute_potential_gradient(cmesh, BX, dhat)
        gB = cmesh.to_full_dof(gB)

        # Cannot compute gradient without barrier stiffness
        if self.params.kB is None:
            binit = BarrierInitializer(self.params)
            binit(x, gU, gB)

        kB = self.params.kB
        BXdot = to_surface(v, mesh, cmesh)
        fconstraints.build(cmesh, BX, cconstraints, dhat, kB, mu)
        gF = fconstraints.compute_potential_gradient(cmesh, BXdot, epsv)
        gF = cmesh.to_full_dof(gF)
        g = M @ (x - xtilde) + dt2*gU + kB * gB + dt*gF
        return g

class Hessian():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray) -> sp.sparse.csc_matrix:
        dt = self.params.dt
        dt2 = self.params.dt2
        xt = self.params.xt
        M = self.params.M
        hep = self.params.hep
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        cconstraints = self.params.cconstraints
        fconstraints = self.params.fconstraints
        dhat = self.params.dhat
        dmin = self.params.dmin
        mu = self.params.mu
        epsv = self.params.epsv
        kB = self.params.kB

        hep.compute_element_elasticity(x, grad=False, hessian=True)
        HU = hep.hessian()
        v = (x - xt) / dt
        BX = to_surface(x, mesh, cmesh)
        BXdot = to_surface(v, mesh, cmesh)
        HB = cconstraints.compute_potential_hessian(
            cmesh, BX, dhat, project_hessian_to_psd=True)
        HB = cmesh.to_full_dof(HB)
        HF = fconstraints.compute_potential_hessian(
            cmesh, BXdot, epsv, project_hessian_to_psd=True)
        HF = cmesh.to_full_dof(HF)
        H = M + dt2*HU + kB * HB + dt*HF
        return H

class LinearSolver():

    def __init__(self, dofs: np.ndarray):
        self.dofs = dofs

    def __call__(self, A: sp.sparse.csc_matrix, b: np.ndarray) -> np.ndarray:
        dofs = self.dofs
        Add = A.tocsr()[dofs, :].tocsc()[:, dofs]
        bd = b[dofs]
        Addinv = pbat.math.linalg.ldlt(Add)
        Addinv.compute(Add)
        # NOTE: If built from source with SuiteSparse, use faster chol
        # Addinv = pbat.math.linalg.chol(
        #     Add, solver=pbat.math.linalg.SolverBackend.SuiteSparse)
        # Addinv.compute(sp.sparse.tril(
        #     Add), pbat.math.linalg.Cholmod.SparseStorage.SymmetricLowerTriangular)
        x = np.zeros_like(b)
        x[dofs] = Addinv.solve(bd).squeeze()
        return x

class CCD():
    def __init__(self,
                 params: Parameters,
                 broad_phase_method: ipctk.BroadPhaseMethod = ipctk.BroadPhaseMethod.HASH_GRID):
        self.params = params
        self.broad_phase_method = broad_phase_method

    def __call__(self, x: np.ndarray, dx: np.ndarray) -> float:
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dmin = self.params.dmin
        broad_phase_method = self.broad_phase_method

        BXt0 = to_surface(x, mesh, cmesh)
        BXt1 = to_surface(x + dx, mesh, cmesh)
        max_alpha = ipctk.compute_collision_free_stepsize(
            cmesh,
            BXt0,
            BXt1,
            broad_phase_method=broad_phase_method,
            min_distance=dmin
        )
        return max_alpha

class BarrierInitializer():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, x: np.ndarray, gU: np.ndarray, gB: np.ndarray):
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        dhat = self.params.dhat
        dmin = self.params.dmin
        avgmass = self.params.avgmass
        bboxdiag = self.params.bboxdiag
        cconstraints = self.params.cconstraints

        # Compute adaptive barrier stiffness
        BX = to_surface(x, mesh, cmesh)
        kB, maxkB = ipctk.initial_barrier_stiffness(
            bboxdiag, dhat, avgmass, gU, gB, dmin=dmin)
        dprev = cconstraints.compute_minimum_distance(cmesh, BX)
        self.params.kB = kB
        self.params.maxkB = maxkB
        self.params.dprev = dprev

class BarrierUpdater():
    def __init__(self, params: Parameters):
        self.params = params

    def __call__(self, xk: np.ndarray):
        mesh = self.params.mesh
        cmesh = self.params.cmesh
        kB = self.params.kB
        maxkB = self.params.maxkB
        dprev = self.params.dprev
        bboxdiag = self.params.bboxdiag
        dhat = self.params.dhat
        dmin = self.params.dmin
        cconstraints = self.params.cconstraints

        BX = to_surface(xk, mesh, cmesh)
        dcurrent = cconstraints.compute_minimum_distance(cmesh, BX)
        self.params.kB = ipctk.update_barrier_stiffness(
            dprev, dcurrent, maxkB, kB, bboxdiag, dmin=dmin)
        self.params.dprev = dcurrent


class RedisClient:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: str = None):
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db, password=password, decode_responses=True)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('simulation_commands')  # Subscribe to the simulation_commands channel
        self.command = None
        self.lock = threading.Lock()
        self.simulation_status = "stopped"  # Track the status of the simulation
        self.listener_thread = threading.Thread(target=self.listen_commands, daemon=True)
        self.listener_thread.start()
        logging.info("RedisClient initialized and listener thread started.")

    def listen_commands(self):
        logging.info("Listener thread started, waiting for simulation commands.")
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                command = message['data'].strip().lower()
                logging.info(f"Received command from Redis: {command}")
                with self.lock:
                    self.command = command
                # Optionally, handle the command immediately or leave it to the main loop

    def get_command(self):
        with self.lock:
            cmd = self.command
            self.command = None
        return cmd

    def set_data(self, key: str, data: str):
        try:
            logging.info(f"Storing data in Redis with key: {key}")
            self.redis_client.set(key, data)
            logging.debug(f"Data stored successfully for key: {key}")
        except redis.RedisError as e:
            logging.error(f"Failed to set data in Redis for key {key}: {e}")

    def get_data(self, key: str):
        try:
            logging.info(f"Retrieving data from Redis with key: {key}")
            data = self.redis_client.get(key)
            logging.debug(f"Data retrieved for key {key}: {data}")
            return data
        except redis.RedisError as e:
            logging.error(f"Failed to get data from Redis for key {key}: {e}")
            return None

    def publish_data(self, channel: str, data: str):
        try:
            logging.info(f"Publishing data to channel: {channel}")
            self.redis_client.publish(channel, data)
            logging.debug(f"Data published successfully to channel: {channel}")
        except redis.RedisError as e:
            logging.error(f"Failed to publish data to channel {channel}: {e}")

    def serialize_mesh_data(self, mesh_data: dict) -> str:
        try:
            logging.info("Serializing mesh data.")
            mesh_data_bson = bson.dumps(mesh_data)
            mesh_data_compressed = zlib.compress(mesh_data_bson)
            mesh_data_b64 = base64.b64encode(mesh_data_compressed).decode('utf-8')
            logging.debug("Mesh data serialized successfully.")
            return mesh_data_b64
        except Exception as e:
            logging.error(f"Failed to serialize mesh data: {e}")
            return None

    # Command methods
    def start(self):
        with self.lock:
            if self.simulation_status == "stopped":
                self.simulation_status = "running"
                logging.info("Starting simulation.")
                self.publish_data('simulation_commands', 'start')
            else:
                logging.warning("Simulation already running or paused.")

    def stop(self):
        with self.lock:
            if self.simulation_status in ["running", "paused"]:
                self.simulation_status = "stopped"
                logging.info("Stopping simulation.")
                self.publish_data('simulation_commands', 'stop')
            else:
                logging.warning("Simulation is not running.")

    def pause(self):
        with self.lock:
            if self.simulation_status == "running":
                self.simulation_status = "paused"
                logging.info("Pausing simulation.")
                self.publish_data('simulation_commands', 'pause')
            else:
                logging.warning("Cannot pause; simulation is not running.")

    def resume(self):
        with self.lock:
            if self.simulation_status == "paused":
                self.simulation_status = "running"
                logging.info("Resuming simulation.")
                self.publish_data('simulation_commands', 'resume')
            else:
                logging.warning("Cannot resume; simulation is not paused.")

    def play(self):
        with self.lock:
            if self.simulation_status == "paused":
                self.simulation_status = "running"
                logging.info("Resuming simulation.")
                self.publish_data('simulation_commands', 'play')
            else:
                logging.warning("Cannot play; simulation is not paused.")

    def kill(self):
        logging.info("Killing the simulation.")
        self.redis_client.publish('simulation_commands', 'kill')

# Simulation loop
def run_simulation(
    mesh,
    x,
    v,
    a,
    M,
    hep,
    dt,
    cmesh,
    cconstraints,
    fconstraints,
    dhat,
    dmin,
    mu,
    epsv,
    dofs,
    redis_client=RedisClient()
):
    max_iters = 100000  # Number of simulation steps
    allowed_commands = {"start", "pause", "stop", "resume", "play", "kill"}
    simulation_running = False  # Track if the simulation is active

    # Precompute objects that do not change within the loop
    solver = LinearSolver(dofs)

    for i in range(max_iters):
        # Retrieve the latest command from Redis
        simulation_command = redis_client.get_command()

        if simulation_command:
            if simulation_command in allowed_commands:
                if simulation_command == "start":
                    if not simulation_running:
                        simulation_running = True
                        logging.info("Simulation started.")
                    else:
                        logging.warning("Simulation is already running.")
                elif simulation_command == "pause":
                    if simulation_running:
                        simulation_running = False
                        logging.info("Simulation paused.")
                    else:
                        logging.warning("Simulation is not running; cannot pause.")
                elif simulation_command == "play":
                    if not simulation_running:
                        simulation_running = True
                        logging.info("Resuming simulation.")
                    else:
                        logging.warning("Simulation is already running.")
                elif simulation_command == "stop":
                    logging.info("Stopping simulation.")
                    simulation_running = False
                elif simulation_command == "kill":
                    logging.info("Killing simulation.")
                    sys.exit()

            else:
                logging.warning(f"Invalid simulation command received: {simulation_command}")

        if not simulation_running:
            # If simulation is not running, skip the simulation step
            logging.debug("Simulation is paused or not started. Waiting for commands.")
            time.sleep(0.1)  # Sleep briefly to prevent tight loop
            continue

        # Set up simulation parameters
        params = Parameters(
            mesh=mesh,
            xt=x,
            vt=v,
            a=a,
            M=M,
            hep=hep,
            dt=dt,
            cmesh=cmesh,
            cconstraints=cconstraints,
            fconstraints=fconstraints,
            dhat=dhat,
            dmin=dmin,
            mu=mu,
            epsv=epsv,
        )

        # Potential, Gradient, and Hessian objects
        f = Potential(params)
        g = Gradient(params)
        H = Hessian(params)

        # Barrier Updater and CCD
        updater = BarrierUpdater(params)
        ccd = CCD(params)

        # Run Newton's method to compute the next time step
        xtp1 = newton(
            x,
            f,
            g,
            H,
            solver,
            alpha0=ccd,
            maxiters=10,
            rtol=1e-5,
            callback=updater
        )

        # Update velocities and positions
        v = (xtp1 - x) / dt
        x = xtp1

        # Map the updated positions to the collision mesh
        BX = to_surface(x, mesh, cmesh)

        mesh_data = {
            "timestamp": time.time(),
            "step": i,
            "x": x.tobytes(),
            "x_shape": x.shape,
            "x_dtype": str(x.dtype),
            "BX": BX.tobytes(),
            "BX_shape": BX.shape,
            "BX_dtype": str(BX.dtype),
            "faces": cmesh.faces.tobytes(),
            "faces_shape": cmesh.faces.shape,
            "faces_dtype": str(cmesh.faces.dtype)
        }

        # Serialize the mesh data
        mesh_data_b64 = redis_client.serialize_mesh_data(mesh_data)

        if mesh_data_b64:
            # Store mesh state in Redis
            redis_client.set_data('mesh_state', mesh_data_b64)

            # Publish mesh data to Redis channel
            logging.info(f"Step {i}: Publishing mesh data to 'simulation_updates' channel.")
            redis_client.publish_data('simulation_updates', mesh_data_b64)

# Main server function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="3D elastic simulation of linear FEM tetrahedra using Incremental Potential Contact",
    )
    parser.add_argument(
        "-i", "--input", help="Path to input mesh", dest="input", required=True
    )
    parser.add_argument(
        "--percent-fixed",
        help="Percentage of input mesh's bottom to fix",
        type=float,
        dest="percent_fixed",
        default=0.1,
    )
    parser.add_argument(
        "-m",
        "--mass-density",
        help="Mass density",
        type=float,
        dest="rho",
        default=1000.0,
    )
    parser.add_argument(
        "-Y",
        "--young-modulus",
        help="Young's modulus",
        type=float,
        dest="Y",
        default=1e6,
    )
    parser.add_argument(
        "-n",
        "--poisson-ratio",
        help="Poisson's ratio",
        type=float,
        dest="nu",
        default=0.45,
    )
    parser.add_argument(
        "-c",
        "--copy",
        help="Number of copies of input model",
        type=int,
        dest="ncopy",
        default=1,
    )
    parser.add_argument(
        "--redis-host",
        help="Redis host address",
        type=str,
        dest="redis_host",
        default="localhost",
    )
    parser.add_argument(
        "--redis-port",
        help="Redis port",
        type=int,
        dest="redis_port",
        default=6379,
    )
    parser.add_argument(
        "--redis-db",
        help="Redis database",
        type=int,
        dest="redis_db",
        default=0,
    )
    args = parser.parse_args()

    # Load input meshes and combine them into 1 mesh
    V, C = [], []
    imesh = meshio.read(args.input)
    V1 = imesh.points.astype(np.float64, order='C')
    C1 = imesh.cells_dict["tetra"].astype(np.int64, order='C')
    V.append(V1)
    C.append(C1)
    for c in range(args.ncopy):
        R = sp.spatial.transform.Rotation.from_quat(
            [0, 0, np.sin(np.pi/4), np.cos(np.pi/4)]).as_matrix()
        V2 = (V[-1] - V[-1].mean(axis=0)) @ R.T + V[-1].mean(axis=0)
        V2[:, 2] += (V2[:, 2].max() - V2[:, 2].min()) + 20
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

    # Mass matrix and external forces (e.g., gravity)
    detJeM = pbat.fem.jacobian_determinants(mesh, quadrature_order=2)
    rho = args.rho
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
    fe = np.tile(rho * g_vec[:, np.newaxis], mesh.E.shape[1])
    f_ext = fe @ Qf @ Nf
    f_ext = f_ext.reshape(math.prod(f_ext.shape), order="F")
    a = Minv @ f_ext

    # Hyperelastic potential
    detJeU = pbat.fem.jacobian_determinants(mesh, quadrature_order=1)
    GNeU = pbat.fem.shape_function_gradients(mesh, quadrature_order=1)
    Y = np.full(mesh.E.shape[1], args.Y)
    nu = np.full(mesh.E.shape[1], args.nu)
    psi = pbat.fem.HyperElasticEnergy.StableNeoHookean
    hep = pbat.fem.HyperElasticPotential(
        mesh, detJeU, GNeU, Y, nu, energy=psi, quadrature_order=1
    )
    hep.precompute_hessian_sparsity()

    # IPC contact handling
    F = igl.boundary_facets(C)
    E = ipctk.edges(F)
    cmesh = ipctk.CollisionMesh.build_from_full_mesh(V, E, F)
    cconstraints = ipctk.CollisionConstraints()
    fconstraints = ipctk.FrictionConstraints()

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

    # Run the simulation
    run_simulation(
        mesh,
        x,
        v,
        a,
        M,
        hep,
        dt=0.01,
        cmesh=cmesh,
        cconstraints=cconstraints,
        fconstraints=fconstraints,
        dhat=dhat,
        dmin=dmin,
        mu=mu,
        epsv=epsv,
        dofs=dofs,
        redis_client=RedisClient(host=args.redis_host, port=args.redis_port, db=args.redis_db)
    )
