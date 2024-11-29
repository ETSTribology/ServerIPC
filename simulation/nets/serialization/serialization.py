# solvers/serialization.py

import base64
import logging
import pickle
from typing import Dict, Optional

import bson
from nets.serialization.serializer import Serializer

logger = logging.getLogger(__name__)


# Pickle Serializer
class PickleSerializer(Serializer):
    def serialize(self, data: Dict) -> Optional[str]:
        try:
            serialized = pickle.dumps(data)
            encoded = base64.b64encode(serialized).decode("utf-8")
            logger.debug("Pickle serialization successful.")
            return encoded
        except Exception as e:
            logger.error(f"Pickle serialization failed: {e}")
            return None

    def deserialize(self, data_str: str) -> Optional[Dict]:
        try:
            serialized = base64.b64decode(data_str)
            data = pickle.loads(serialized)
            logger.debug("Pickle deserialization successful.")
            return data
        except Exception as e:
            logger.error(f"Pickle deserialization failed: {e}")
            return None


# JSON Serializer
class JSONSerializer(Serializer):
    def serialize(self, data: Dict) -> Optional[str]:
        import json

        try:
            serialized = json.dumps(data)
            logger.debug("JSON serialization successful.")
            return serialized
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            return None

    def deserialize(self, data_str: str) -> Optional[Dict]:
        import json

        try:
            data = json.loads(data_str)
            logger.debug("JSON deserialization successful.")
            return data
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            return None


# BSON Serializer
class BSONSerializer(Serializer):
    def serialize(self, data: Dict) -> Optional[str]:
        try:
            serialized = bson.BSON.encode(data)
            encoded = base64.b64encode(serialized).decode("utf-8")
            logger.debug("BSON serialization successful.")
            return encoded
        except ImportError:
            logger.error(
                "BSON library not installed. Install it using 'pip install bson'."
            )
            return None
        except Exception as e:
            logger.error(f"BSON serialization failed: {e}")
            return None

    def deserialize(self, data_str: str) -> Optional[Dict]:
        try:
            serialized = base64.b64decode(data_str)
            data = BSON(serialized).decode()
            logger.debug("BSON deserialization successful.")
            return data
        except ImportError:
            logger.error(
                "BSON library not installed. Install it using 'pip install bson'."
            )
            return None
        except Exception as e:
            logger.error(f"BSON deserialization failed: {e}")
            return None
