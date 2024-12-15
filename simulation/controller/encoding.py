import base64
import json
from enum import Enum


class EncodingType(Enum):
    """Supported encoding types for requests/responses"""
    JSON = "json"
    BASE64 = "base64"
    PLAIN = "plain"

    @staticmethod
    def from_str(encoding: str) -> "EncodingType":
        """Convert string to EncodingType"""
        encoding = encoding.lower()
        if encoding in EncodingType.__members__:
            return EncodingType[encoding.upper()]
        return EncodingType.JSON

def encode_parameters(parameters: dict, encoding: EncodingType) -> dict:
    """Encode parameters based on the specified encoding type."""
    data = parameters.copy()
    if encoding == EncodingType.BASE64:
        data["parameters"] = base64.b64encode(json.dumps(data["parameters"]).encode()).decode()
    return data

def decode_parameters(data: dict) -> dict:
    """Decode parameters if they are base64 encoded."""
    encoding = EncodingType.from_str(data.get("encoding", "json"))
    if encoding == EncodingType.BASE64:
        params_b64 = data.get("parameters", "")
        try:
            params_json = base64.b64decode(params_b64).decode()
            data["parameters"] = json.loads(params_json)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 parameters: {e}")
    return data