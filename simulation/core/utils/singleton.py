from typing import Any, Dict, Type


class SingletonMeta(type):
    """A Singleton metaclass that ensures only one instance of a class exists."""

    _instances: Dict[Type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
