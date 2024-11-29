# SurrealDBHandler Log Messages
SURREALDB_HANDLER_INITIALIZED = "SurrealDBHandler initialized for table '%s'."
SURREALDB_EMIT_SUCCESS = "Log record emitted to SurrealDB: %s"
SURREALDB_EMIT_FAILURE = "Failed to emit log record to SurrealDB: %s"

# MinIOHandler Log Messages
MINIO_HANDLER_INITIALIZED = "MinIOHandler initialized for bucket '%s' and object '%s'."
MINIO_BUCKET_CREATED = "MinIO bucket '%s' created."
MINIO_BUCKET_EXISTS = "MinIO bucket '%s' already exists."
MINIO_FLUSH_UPLOAD = "Uploaded %d log records to MinIO as '%s'."
MINIO_FLUSH_FAILURE = "Failed to upload log records to MinIO: %s"

# HandlerFactory Log Messages
HANDLER_FACTORY_INITIALIZED = (
    "HandlerFactory initialized with provided HandlerRegistry."
)
HANDLER_FACTORY_CREATION_SUCCESS = "Created handler '%s' with config: %s"
HANDLER_FACTORY_CREATION_FAILURE = "Failed to create handler '%s': %s"

# LoggingManager Log Messages
LOGGING_MANAGER_REMOVE_HANDLER = "Removed an existing handler from the root logger."
LOGGING_MANAGER_SETUP_DEFAULT = "Default console logging has been set up."
LOGGING_MANAGER_SET_LEVEL = "Set root logger level to '%s'."
LOGGING_MANAGER_SETUP_FORMATTER = "Formatter '%s' has been set up."
LOGGING_MANAGER_ADD_HANDLER = "Added handler '%s' of type '%s' to the root logger."
LOGGING_MANAGER_FAILED_CONFIGURE_HANDLER = "Failed to configure handler '%s': %s"
LOGGING_MANAGER_LOADED_CONFIG = "Loaded logging configuration from '%s'."
