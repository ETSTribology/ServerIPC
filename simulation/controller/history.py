# history.py
from dataclasses import dataclass
from typing import List
from threading import Lock

@dataclass
class CommandHistoryEntry:
    timestamp: str
    command_name: str
    request_id: str
    status: str
    message: str


class CommandHistory:
    def __init__(self):
        self.history: List[CommandHistoryEntry] = []
        self._lock = Lock()

    def add_entry(self, entry: CommandHistoryEntry):
        with self._lock:
            self.history.append(entry)

    def get_history(self) -> List[CommandHistoryEntry]:
        with self._lock:
            return list(self.history)