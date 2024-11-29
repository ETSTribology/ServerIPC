from nets.serialization.serialization import (
    BSONSerializer,
    JSONSerializer,
    PickleSerializer,
    Serializer,
)


class SerializerFactory:
    _serializers = {
        "pickle": PickleSerializer,
        "json": JSONSerializer,
        "bson": BSONSerializer,
    }

    @classmethod
    def get_serializer(cls, method: str) -> Serializer:
        method = method.lower()
        serializer_class = cls._serializers.get(method)
        if not serializer_class:
            raise ValueError(
                f"Unsupported serialization method: '{method}'. Supported methods are: {', '.join(cls._serializers.keys())}."
            )
        return serializer_class()
