import argparse
import logging

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="3D Elastic Simulation of Linear FEM Tetrahedra using IPC",
        description="Simulate 3D elastic deformations with contact handling."
    )
    parser.add_argument("-i", "--input", help="Path to input mesh", required=False)
    parser.add_argument("--percent-fixed", type=float, default=0.0,
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
    parser.add_argument("-j", "--n-threads", type=int, default=4,
                        help="Number of threads to use for parallel Newton solver")
    parser.add_argument("--json", type=str, default=None, help="Path to JSON file with simulation parameters")
    return parser.parse_args()