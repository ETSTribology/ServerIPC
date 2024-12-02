import json
import uuid
from typing import Any, Dict, Optional

import redis


def generate_request_id() -> str:
    return str(uuid.uuid4())


def publish_command(command: str, payload: Optional[Dict[str, Any]] = None):
    r = redis.Redis(host="localhost", port=6379, db=0)

    request = {"request_id": generate_request_id(), "command": command, "payload": payload}

    message = json.dumps(request)

    r.publish("simulation_commands", message)
    print(f"Published command: {message}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Publish simulation commands to Redis.")
    parser.add_argument("command", type=str, help="The command to publish (e.g., start, pause).")
    parser.add_argument(
        "--payload", type=str, default=None, help="Optional JSON payload for the command."
    )

    args = parser.parse_args()

    # Parse payload if provided
    payload = json.loads(args.payload) if args.payload else None

    publish_command(args.command, payload)
