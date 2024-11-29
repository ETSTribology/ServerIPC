import argparse
import logging
from application import ClientApplication

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D FEM Simulation Client with Redis and Polyscope Visualization."
    )
    parser.add_argument(
        "--redis-host", type=str, default="localhost", help="Redis server host"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis server port"
    )
    parser.add_argument(
        "--redis-db", type=int, default=0, help="Redis database number"
    )
    parser.add_argument(
        "--enable-screenshots",
        action="store_true",
        help="Enable screenshot capture",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=1.0,
        help="Interval between screenshots in seconds",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default="screenshots",
        help="Directory to save screenshots",
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    client_app = ClientApplication(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )
    client_app.run()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
