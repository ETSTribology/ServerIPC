import argparse
import logging
from application import ClientApplication

import os

# Set environment variables before importing any libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent multiple OpenMP runtimes
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


def parse_arguments():
    parser = argparse.ArgumentParser(description="3D FEM Simulation Client with Redis and Polyscope Visualization.")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis server host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis server port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database number")
    return parser.parse_args()

def main():
    args = parse_arguments()
    client_app = ClientApplication(redis_host=args.redis_host, redis_port=args.redis_port, redis_db=args.redis_db)
    client_app.run()

if __name__ == "__main__":
    # Configure logging for the client
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
