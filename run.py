# start_simulation.py

import argparse
import os
import signal
import subprocess
import sys


def main():

    parser = argparse.ArgumentParser(
        description="Script to start both simulation server and visualization client simultaneously."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario/rectangle.json",
        help="Path to the scenario JSON file for the simulation server."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="visualization/config.json",
        help="Path to the JSON configuration file for the visualization client."
    )
    args = parser.parse_args()

    # Define the commands for server and client
    server_cmd = ["python3", "simulation/server.py", "--json", args.scenario]
    client_cmd = ["python3", "simulation/visualization/client.py", "--json", args.config]

    try:
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Started server (PID: {server_process.pid})")
    except FileNotFoundError:
        print("Error: 'simulation/server.py' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

    try:
        client_process = subprocess.Popen(
            client_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Started client (PID: {client_process.pid})")
    except FileNotFoundError:
        print("Error: 'simulation/visualization/client.py' not found.")
        server_process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"Failed to start client: {e}")
        server_process.terminate()
        sys.exit(1)

    def terminate_processes(signum, frame):
        print("\nTermination signal received. Shutting down processes...")
        server_process.terminate()
        client_process.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, terminate_processes)
    signal.signal(signal.SIGTERM, terminate_processes)

    try:
        while True:
            server_retcode = server_process.poll()
            client_retcode = client_process.poll()

            if server_retcode is not None:
                print(f"Server terminated with return code {server_retcode}")
                break

            if client_retcode is not None:
                print(f"Client terminated with return code {client_retcode}")
                break

    except KeyboardInterrupt:
        terminate_processes(None, None)

    finally:
        # Ensure both processes are terminated
        if server_process.poll() is None:
            server_process.terminate()
        if client_process.poll() is None:
            client_process.terminate()

if __name__ == "__main__":
    main()
