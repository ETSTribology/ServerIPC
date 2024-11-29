import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Serializer Interface
class Serializer(ABC):
    @abstractmethod
    def serialize(self, data: Dict) -> Optional[str]:
        pass

    @abstractmethod
    def deserialize(self, data_str: str) -> Optional[Dict]:
        pass
