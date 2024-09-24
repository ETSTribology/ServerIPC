import argparse
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
from loop import run_simulation
from utils.mesh_utils import combine, to_surface, find_codim_vertices
from utils.logging_setup import setup_logging
from materials import Material
from core.parameters import Parameters
from args import parse_arguments

from initialization import (
    initialization
)


logger = logging.getLogger(__name__)

ipctk.set_num_threads(10)

def main():
    args, mesh, x, v, a, M, hep, dt, cmesh, cconstraints, fconstraints, dhat, dmin, mu, epsv, dofs, redis_client, material, barrier_potential, friction_potential, n, f_ext, Qf, Nf, qgf, Y, nu, psi, detJeU, GNeU, E, F, aabb, vdbc, dX = initialization()

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
        material=material,
        barrier_potential=barrier_potential,
        friction_potential=friction_potential
    )

if __name__ == "__main__":
    main()
