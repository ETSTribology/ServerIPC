import logging
import time
import sys
from typing import Callable
from solvers.linear_solver import LinearSolver
from contact.barrier_updater import BarrierUpdater
from contact.ccd import CCD
from core.parameters import Parameters
from core.potential import Potential
from core.gradient import Gradient
from core.hessian import Hessian
from redis_interface.redis_client import SimulationRedisClient
from solvers.newton import newton, parallel_newton
import numpy as np
from materials import Material
from utils.mesh_utils import to_surface
import ipctk
from initialization import (
    initialization
)

logger = logging.getLogger(__name__)

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
    redis_client: SimulationRedisClient,
    material: Material = None,
    barrier_potential: ipctk.BarrierPotential = None,
    friction_potential: ipctk.FrictionPotential = None
):
    max_iters = 100000  # Number of simulation steps
    allowed_commands = {"start", "pause", "stop", "resume", "play", "kill", "reset"}
    simulation_running = False  # Track if the simulation is active

    # Precompute objects that do not change within the loop
    solver = LinearSolver(dofs)

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
        material=material,
        dhat=dhat,
        dmin=dmin,
        mu=mu,
        epsv=epsv,
        barrier_potential=barrier_potential,
        friction_potential=friction_potential
    )

    for i in range(max_iters):
        # Retrieve the latest command from Redis
        simulation_command = redis_client.get_command()

        if simulation_command:
            if simulation_command in allowed_commands:
                if simulation_command == "start":
                    if not simulation_running:
                        simulation_running = True
                        logger.info("Simulation started.")
                    else:
                        logger.warning("Simulation is already running.")
                elif simulation_command == "pause":
                    if simulation_running:
                        simulation_running = False
                        logger.info("Simulation paused.")
                    else:
                        logger.warning("Simulation is not running; cannot pause.")
                elif simulation_command in {"resume", "play"}:
                    if not simulation_running:
                        simulation_running = True
                        logger.info("Resuming simulation.")
                    else:
                        logger.warning("Simulation is already running.")
                elif simulation_command == "stop":
                    logger.info("Stopping simulation.")
                    simulation_running = False
                elif simulation_command == "kill":
                    logger.info("Killing simulation.")
                    sys.exit()
                elif simulation_command == "reset":
                    args, mesh, x, v, a, M, hep, dt, cmesh, cconstraints, fconstraints, dhat, dmin, mu, epsv, dofs, redis_client, material, barrier_potential, friction_potential, n, f_ext, Qf, Nf, qgf, Y, nu, psi, detJeU, GNeU, E, F, aabb, vdbc, dX = initialization()
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
                        material=material,
                        dhat=dhat,
                        dmin=dmin,
                        mu=mu,
                        epsv=epsv,
                        barrier_potential=barrier_potential,
                        friction_potential=friction_potential
                    )
                    simulation_running = False
                    logger.info("Simulation reset.")
                    simulation_running = True
            else:
                logger.warning(f"Invalid simulation command received: {simulation_command}")

        if not simulation_running:
            # If simulation is not running, skip the simulation step
            logger.debug("Simulation is paused or not started. Waiting for commands.")
            time.sleep(0.1)  # Sleep briefly to prevent tight loop
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
            n_threads=4  # Adjust based on your system
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
            logger.info(f"Step {i}: Publishing mesh data to 'simulation_updates' channel.")
            redis_client.publish_data('simulation_updates', mesh_data_b64)