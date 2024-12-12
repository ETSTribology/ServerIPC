from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SimulationLogMessageCode(Enum):
    """Enumeration for simulation log message codes with unique identifiers."""
    # Configuration and Initialization Logs
    CONFIGURATION_LOADED = ("Configuration loaded successfully", "LOG_001")
    CONFIGURATION_FAILED = ("Failed to load configuration", "LOG_002")
    STORAGE_INITIALIZED = ("Storage initialized successfully", "LOG_003")
    STORAGE_FAILED = ("Failed to initialize storage", "LOG_004")
    BACKEND_INITIALIZED = ("Backend initialized successfully", "LOG_005")
    BACKEND_FAILED = ("Failed to initialize backend", "LOG_006")
    DATABASE_INITIALIZED = ("Database initialized successfully", "LOG_007")
    DATABASE_FAILED = ("Failed to initialize database", "LOG_008")
    INITIAL_CONDITIONS_SET = ("Initial conditions set up", "LOG_009")

    # Setup and Matrix Logs
    MASS_MATRIX_SETUP = ("Mass matrix set up", "LOG_010")
    MASS_MATRIX_FAILED = ("Failed to set up mass matrix", "LOG_011")
    EXTERNAL_FORCES_SETUP = ("External forces set up", "LOG_012")
    EXTERNAL_FORCES_FAILED = ("Failed to set up external forces", "LOG_013")
    HYPERELASTIC_SETUP = ("Hyperelastic potential set up", "LOG_014")
    HYPERELASTIC_FAILED = ("Failed to set up hyperelastic potential", "LOG_015")
    COLLISION_SETUP = ("Collision mesh set up", "LOG_016")
    COLLISION_FAILED = ("Failed to set up collision mesh", "LOG_017")
    BOUNDARY_CONDITIONS_SETUP = ("Boundary conditions set up", "LOG_018")
    BOUNDARY_CONDITIONS_FAILED = ("Failed to set up boundary conditions", "LOG_019")

    # Simulation State Logs
    SIMULATION_STATE_INITIALIZED = ("Simulation state initialized successfully", "LOG_020")
    SIMULATION_INITIALIZATION_ABORTED = ("Simulation initialization aborted", "LOG_021")
    POTENTIAL_SETUP = ("Potential setup completed", "LOG_022")
    GRADIENT_SETUP = ("Gradient setup completed", "LOG_023")
    HESSIAN_SETUP = ("Hessian setup completed", "LOG_024")
    CCD_SETUP = ("CCD setup completed", "LOG_025")
    SIMULATION_STEP_COMPLETED = ("Simulation step completed", "LOG_026")
    SIMULATION_PAUSED = ("Simulation paused", "LOG_027")
    SIMULATION_RESUMED = ("Simulation resumed", "LOG_028")
    SIMULATION_SHUTDOWN = ("Simulation shut down gracefully", "LOG_029")
    SIMULATION_STARTED = ("Simulation started", "LOG_030")
    SIMULATION_STOPPED = ("Simulation stopped", "LOG_031")

    # Checkpoint and Results Logs
    CHECKPOINT_SAVED = ("Checkpoint saved", "LOG_032")
    CHECKPOINT_FAILED = ("Failed to save checkpoint", "LOG_033")
    RESULTS_SAVED = ("Results saved successfully", "LOG_034")
    RESULTS_SAVE_FAILED = ("Failed to save results", "LOG_035")

    # Error and Warning Logs
    USER_INTERRUPTION = ("Simulation interrupted by user", "LOG_036")
    MEMORY_ALLOCATION_FAILED = ("Memory allocation failed", "LOG_037")
    INVALID_INPUT_DETECTED = ("Invalid input detected", "LOG_038")
    PERFORMANCE_WARNING = ("Performance warning", "LOG_039")
    RESOURCE_LIMIT_REACHED = ("Resource limit reached", "LOG_040")

    # Redis Backend Logs
    REDIS_CONNECTED = ("Connected to Redis backend", "LOG_041")
    REDIS_DISCONNECTED = ("Disconnected from Redis backend", "LOG_042")
    REDIS_CONNECTION_FAILED = ("Failed to connect to Redis", "LOG_043")
    REDIS_DISCONNECTION_FAILED = ("Failed to disconnect from Redis", "LOG_044")
    REDIS_WRITE_SUCCESS = ("Written data to Redis", "LOG_045")
    REDIS_WRITE_FAILURE = ("Failed to write data to Redis", "LOG_046")
    REDIS_READ_SUCCESS = ("Read data from Redis", "LOG_047")
    REDIS_READ_FAILURE = ("Failed to read data from Redis", "LOG_048")

    # WebSocket Backend Logs
    WEBSOCKET_CONNECTED = ("Connected to WebSocket backend", "LOG_049")
    WEBSOCKET_DISCONNECTED = ("Disconnected from WebSocket backend", "LOG_050")
    WEBSOCKET_CONNECTION_FAILED = ("Failed to connect to WebSocket", "LOG_051")
    WEBSOCKET_DISCONNECTION_FAILED = ("Failed to disconnect from WebSocket", "LOG_052")
    WEBSOCKET_WRITE_SUCCESS = ("Written data to WebSocket", "LOG_053")
    WEBSOCKET_WRITE_FAILURE = ("Failed to write data to WebSocket", "LOG_054")
    WEBSOCKET_READ_SUCCESS = ("Read data from WebSocket", "LOG_055")
    WEBSOCKET_READ_FAILURE = ("Failed to read data from WebSocket", "LOG_056")

    # TensorBoard Logs
    TENSORBOARD_INITIALIZED = ("TensorBoard initialized", "LOG_057")
    TENSORBOARD_INITIALIZATION_FAILED = ("Failed to initialize TensorBoard", "LOG_058")
    TENSORBOARD_SCALAR_LOGGED = ("Logged scalar to TensorBoard", "LOG_059")
    TENSORBOARD_TEXT_LOGGED = ("Logged text to TensorBoard", "LOG_060")
    TENSORBOARD_GRAPH_LOGGED = ("Logged graph to TensorBoard", "LOG_061")
    TENSORBOARD_GRAPH_LOGGING_FAILED = ("Failed to log graph to TensorBoard", "LOG_062")

    # Mesh Management Logs
    MESH_LOADED = ("Mesh loaded", "LOG_063")
    MESH_LOAD_FAILED = ("Failed to load mesh", "LOG_064")
    MESH_VALIDATION_FAILED = ("Mesh validation failed", "LOG_065")
    MESH_SAVED = ("Mesh saved", "LOG_066")
    MESH_SAVE_FAILED = ("Failed to save mesh", "LOG_067")

    PARAMETERS_SETUP = ("Parameters Setup Error", "LOG_068")

    # Command Logs
    COMMAND_INITIALIZED = ("Command initialized successfully", "LOG_069")
    COMMAND_STARTED = ("Command started", "LOG_070")
    COMMAND_SUCCESS = ("Command executed successfully", "LOG_071")
    COMMAND_RETRY = ("Retrying command execution", "LOG_072")
    COMMAND_FAILED = ("Command execution failed", "LOG_073")

    # New Log Messages
    DATABASE_CONNECTION_SUCCESS = ("Database connection established successfully", "LOG_074")
    DATABASE_CONNECTION_FAILURE = ("Failed to establish database connection", "LOG_075")
    DATA_VALIDATION_SUCCESS = ("Data validation completed successfully", "LOG_076")
    DATA_VALIDATION_FAILURE = ("Data validation failed", "LOG_077")
    USER_AUTHENTICATION_SUCCESS = ("User authenticated successfully", "LOG_078")
    USER_AUTHENTICATION_FAILURE = ("User authentication failed", "LOG_079")

    def __init__(self, description, code):
        self.description = description
        self.code = code

    def __str__(self):
        return f"{self.description}"

    def details(self, details: str):
        return f"{self.description}: {details}"