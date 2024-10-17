import logging
import time
import sys
from typing import Callable, List
from solvers.linear_solver import LinearSolver
from contact.barrier_updater import BarrierUpdater
from contact.ccd import CCD
from core.parameters import Parameters
from core.potential import Potential
from core.gradient import Gradient
from core.hessian import Hessian
from net_interface.redis_client import SimulationRedisClient
from solvers.newton import newton, parallel_newton
import numpy as np
from materials import Material
from utils.mesh_utils import to_surface
import ipctk
from initialization import initialization
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def reset_simulation():
    return initialization()

def run_simulation(
    mesh: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    M: np.ndarray,
    hep: float,
    dt: float,
    cmesh,
    cconstraints,
    fconstraints,
    dhat: float,
    dmin: float,
    mu: float,
    epsv: float,
    dofs: int,
    redis_client: SimulationRedisClient,
    materials: List[Material],
    barrier_potential: ipctk.BarrierPotential = None,
    friction_potential: ipctk.FrictionPotential = None,
    config: dict = None,
    element_materials: List = None,
    num_nodes_list: List = None,
    face_materials: np.ndarray = None,
    instances: Optional[List[Dict]] = None
) -> None:
    material = materials[0]
    max_iters = 100000  # Number of simulation steps
    allowed_commands = {"start", "pause", "stop", "resume", "play", "kill", "reset"}
    simulation_running = False  # Track if the simulation is active

    logger.info("Simulation initialized.")

    # Precompute objects that do not change within the loop
    solver = LinearSolver(dofs, solver_type="cg")

    logger.info("Simulation started.")

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
        materials=materials,
        element_materials=element_materials,
        dhat=dhat,
        dmin=dmin,
        mu=mu,
        epsv=epsv,
        barrier_potential=barrier_potential,
        friction_potential=friction_potential
    )

    logger.info("Simulation parameters set.")

    for i in range(max_iters):
        # Retrieve the latest command from Redis
        simulation_command = redis_client.get_command()

        if simulation_command:
            time.sleep(0.1)  # Sleep briefly to prevent rapid polling

            if simulation_command not in allowed_commands:
                logger.warning(f"Invalid simulation command received: {simulation_command}")
            else:
                if simulation_command in {"start", "resume", "play"}:
                    if not simulation_running:
                        simulation_running = True
                        logger.info(f"Simulation {simulation_command}.")
                    else:
                        logger.warning("Simulation is already running.")
                elif simulation_command == "pause":
                    if simulation_running:
                        simulation_running = False
                        logger.info("Simulation paused.")
                    else:
                        logger.warning("Simulation is not running; cannot pause.")
                elif simulation_command == "stop":
                    logger.info("Stopping simulation.")
                    simulation_running = False
                elif simulation_command == "kill":
                    logger.info("Killing simulation.")
                    sys.exit()
                elif simulation_command == "reset":
                    config, mesh, x, v, a, M, hep, dt, cmesh, cconstraints, fconstraints, dhat, dmin, mu, epsv, dofs, redis_client, materials, barrier_potential, friction_potential, n, f_ext, Qf, Nf, qgf, Y_array, nu_array, psi, detJeU, GNeU, E, F, element_materials, num_nodes_list, face_materials, instances = initialization()
                    # Initialize Parameters instance
                    run_simulation(
                        mesh=mesh,
                        x=x,
                        v=v,
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
                        dofs=dofs,
                        redis_client=redis_client,
                        materials=materials,
                        barrier_potential=barrier_potential,
                        friction_potential=friction_potential,
                        config=config,
                        element_materials=element_materials,
                        num_nodes_list=num_nodes_list,
                        face_materials=face_materials,
                    )

        if not simulation_running:
            # If simulation is not running, skip the simulation step
            try:
                logger.debug("Simulation is paused or not started. Waiting for commands.")
                time.sleep(2)
            except KeyboardInterrupt:
                logger.info("Simulation stopped by user.")
                sys.exit()
            continue

        # Potential, Gradient, and Hessian objects
        f = Potential(params)
        g = Gradient(params)
        H = Hessian(params)

        # Barrier Updater and CCD
        updater = BarrierUpdater(params)
        ccd = CCD(params)

        # Run Newton's method to compute the next time step
        xtp1 = parallel_newton(
            x,
            f,
            g,
            H,
            solver,
            alpha0=ccd,
            maxiters=10,
            rtol=1e-5,
            callback=updater,
            n_threads=8
        )

        # Update velocities and positions
        v = (xtp1 - x) / dt
        x = xtp1

        params.xt = x
        params.vt = v
        params.a = a
        params.xtilde = x + dt * v + params.dt2 * a

        # Map the updated positions to the collision mesh
        BX = to_surface(x, mesh, cmesh)

        mesh_data = {
            "timestamp": time.time(),  # Current timestamp
            "step": i,  # Current simulation step
            "x": x.tobytes(),  # Serialize the positions
            "x_shape": x.shape,  # Shape of the positions array
            "x_dtype": str(x.dtype),  # Data type of the positions array
            "BX": BX.tobytes(),  # Serialize the surface mesh
            "BX_shape": BX.shape,  # Shape of the surface mesh
            "BX_dtype": str(BX.dtype),  # Data type of the surface mesh
            "faces": cmesh.faces.tobytes(),  # Serialize the collision mesh faces
            "faces_shape": cmesh.faces.shape,  # Shape of the collision mesh faces
            "faces_dtype": str(cmesh.faces.dtype),  # Data type of the collision mesh faces
            "face_materials": face_materials.tobytes(),  # Serialize the face materials
            "face_materials_shape": face_materials.shape,  # Shape of the face materials array
            "face_materials_dtype": str(face_materials.dtype),  # Data type of the face materials array
            "materials": materials  # List of materials
        }

        # Serialize the mesh data
        mesh_data_b64 = redis_client.serialize_mesh_data(mesh_data)

        if mesh_data_b64:
            # Store mesh state in Redis
            redis_client.set_data('mesh_state', mesh_data_b64)

            # Publish mesh data to Redis channel
            logger.info(f"Step {i}: Publishing mesh data to 'simulation_updates' channel.")
            redis_client.publish_data('simulation_updates', mesh_data_b64)
