from enum import Enum


class SimulationErrorCode(Enum):
    """Enumeration for simulation error codes with unique identifiers."""
    CONFIGURATION = ("Configuration Error", "ERR_001")
    STORAGE_INITIALIZATION = ("Storage Initialization Error", "ERR_002")
    BACKEND_INITIALIZATION = ("Backend Initialization Error", "ERR_003")
    DATABASE_INITIALIZATION = ("Database Initialization Error", "ERR_004")
    MESH_SETUP = ("Mesh Setup Error", "ERR_005")
    MASS_MATRIX_SETUP = ("Mass Matrix Setup Error", "ERR_006")
    EXTERNAL_FORCES_SETUP = ("External Forces Setup Error", "ERR_007")
    HYPERELASTIC_SETUP = ("Hyperelastic Potential Setup Error", "ERR_008")
    COLLISION_SETUP = ("Collision Setup Error", "ERR_009")
    BOUNDARY_CONDITIONS_SETUP = ("Boundary Conditions Setup Error", "ERR_010")
    LINE_SEARCH_SETUP = ("Line Search Setup Error", "ERR_011")
    LINEAR_SOLVER_SETUP = ("Linear Solver Setup Error", "ERR_012")
    OPTIMIZER_SETUP = ("Optimizer Setup Error", "ERR_013")
    PARAMETERS_SETUP = ("Parameters Setup Error", "ERR_014")
    POTENTIAL_SETUP = ("Potential Setup Error", "ERR_015")
    GRADIENT_SETUP = ("Gradient Setup Error", "ERR_016")
    HESSIAN_SETUP = ("Hessian Setup Error", "ERR_017")
    BARRIER_UPDATER_SETUP = ("Barrier Updater Setup Error", "ERR_018")
    CCD_SETUP = ("CCD Setup Error", "ERR_019")
    SIMULATION_LOOP = ("Simulation Loop Error", "ERR_020")
    COMMAND_PROCESSING = ("Command Processing Error", "ERR_021")
    GENERAL = ("General Error", "ERR_022")
    MEMORY_ALLOCATION = ("Memory Allocation Error", "ERR_023")
    FILE_IO = ("File I/O Error", "ERR_024")
    NETWORK_COMMUNICATION = ("Network Communication Error", "ERR_025")
    THREADING = ("Threading Error", "ERR_026")
    GPU_INITIALIZATION = ("GPU Initialization Error", "ERR_027")
    GPU_COMPUTATION = ("GPU Computation Error", "ERR_028")
    INPUT_VALIDATION = ("Input Validation Error", "ERR_029")
    RESOURCE_ALLOCATION = ("Resource Allocation Error", "ERR_030")
    TIMING_CONSTRAINT = ("Timing Constraint Error", "ERR_031")
    LINE_SEARCH = ("Line Search Setup Error", "ERR_032")
    GRADIENT_CALCULATION = ("Gradient Calculation Error", "ERR_033")
    BACKEND_ERROR = ("Backend Error", "ERR_034")
    NETWORK_ERROR = ("Network Error", "ERR_035")
    FILE_ERROR = ("File Error", "ERR_036")
    MEMORY_ERROR = ("Memory Error", "ERR_037")
    CPU_ERROR = ("CPU Error", "ERR_038")
    BOARD_ERROR = ("Board Error", "ERR_039")
    SERVER_ERROR = ("Server Error", "ERR_040")
    CONNECTION_ERROR = ("Connection Error", "ERR_041")
    CONFIG_ERROR = ("Configuration Error", "ERR_042")
    CORE_ERROR = ("Core Error", "ERR_043")
    MATH_ERROR = ("Math Error", "ERR_044")
    STATE_ERROR = ("State Error", "ERR_045")
    CONTROLLER_ERROR = ("Controller Error", "ERR_046")
    COMMAND_ERROR = ("Command Error", "ERR_047")
    HISTORY_ERROR = ("History Error", "ERR_048")
    EXTENSION_ERROR = ("Extension Error", "ERR_049")
    STORAGE_ERROR = ("Storage Error", "ERR_050")
    IO_ERROR = ("I/O Error", "ERR_051")
    LOGS_ERROR = ("Logs Error", "ERR_052")
    INITIALIZATION_ERROR = ("Initialization Error", "ERR_053")
    MANAGER_ERROR = ("Manager Error", "ERR_054")
    PYTHON_ERROR = ("Python Error", "ERR_055")

    def __init__(self, description, code):
        self.description = description
        self.code = code

    def __str__(self):
        return f"{self.code}: {self.description}"


class SimulationError(Exception):
    """Base class for simulation errors."""

    def __init__(self, code: SimulationErrorCode, message: str, details: str = None):
        """
        Initialize a SimulationError with a code, message, and optional details.

        Args:
            code (SimulationErrorCode): The error code indicating the type of error.
            message (str): A brief description of the error.
            details (str, optional): Additional information or context about the error.
        """
        self.code = code
        self.message = message
        self.details = details
        super().__init__(f"[{code.code}] {message} - {code.description}" + (f" Details: {details}" if details else ""))


def get_simulation_error_class(code: SimulationErrorCode):
    """Return the appropriate error class for a given error code."""
    error_classes = {
        SimulationErrorCode.CONFIGURATION: ConfigurationError,
        SimulationErrorCode.STORAGE_INITIALIZATION: StorageInitializationError,
        SimulationErrorCode.BACKEND_INITIALIZATION: BackendInitializationError,
        SimulationErrorCode.DATABASE_INITIALIZATION: DatabaseInitializationError,
        SimulationErrorCode.MESH_SETUP: MeshSetupError,
        SimulationErrorCode.MASS_MATRIX_SETUP: MassMatrixSetupError,
        SimulationErrorCode.EXTERNAL_FORCES_SETUP: ExternalForcesSetupError,
        SimulationErrorCode.HYPERELASTIC_SETUP: HyperElasticSetupError,
        SimulationErrorCode.COLLISION_SETUP: CollisionSetupError,
        SimulationErrorCode.BOUNDARY_CONDITIONS_SETUP: BoundaryConditionsSetupError,
        SimulationErrorCode.LINE_SEARCH_SETUP: LineSearchSetupError,
        SimulationErrorCode.LINE_SEARCH: LineSearchError,
        SimulationErrorCode.GRADIENT_SETUP: GradientSetupError,
        SimulationErrorCode.GRADIENT_CALCULATION: GradientCalculationError,
        SimulationErrorCode.HESSIAN_SETUP: HessianSetupError,
        SimulationErrorCode.BARRIER_UPDATER_SETUP: BarrierUpdaterSetupError,
        SimulationErrorCode.CCD_SETUP: CCDSetupError,
        SimulationErrorCode.SIMULATION_LOOP: SimulationLoopError,
        SimulationErrorCode.COMMAND_PROCESSING: CommandProcessingError,
        SimulationErrorCode.MEMORY_ALLOCATION: MemoryAllocationError,
        SimulationErrorCode.FILE_IO: FileIOError,
        SimulationErrorCode.NETWORK_COMMUNICATION: NetworkCommunicationError,
        SimulationErrorCode.THREADING: ThreadingError,
        SimulationErrorCode.GPU_INITIALIZATION: GPUInitializationError,
        SimulationErrorCode.GPU_COMPUTATION: GPUComputationError,
        SimulationErrorCode.INPUT_VALIDATION: InputValidationError,
        SimulationErrorCode.RESOURCE_ALLOCATION: ResourceAllocationError,
        SimulationErrorCode.TIMING_CONSTRAINT: TimingConstraintError,
        SimulationErrorCode.GENERAL: GeneralSimulationError,
        SimulationErrorCode.BACKEND_ERROR: BackendError,
        SimulationErrorCode.NETWORK_ERROR: NetworkError,
        SimulationErrorCode.FILE_ERROR: FileError,
        SimulationErrorCode.MEMORY_ERROR: MemoryError,
        SimulationErrorCode.CPU_ERROR: CPUError,
        SimulationErrorCode.BOARD_ERROR: BoardError,
        SimulationErrorCode.SERVER_ERROR: ServerError,
        SimulationErrorCode.CONNECTION_ERROR: ConnectionError,
        SimulationErrorCode.CONFIG_ERROR: ConfigError,
        SimulationErrorCode.CORE_ERROR: CoreError,
        SimulationErrorCode.MATH_ERROR: MathError,
        SimulationErrorCode.STATE_ERROR: StateError,
        SimulationErrorCode.CONTROLLER_ERROR: ControllerError,
        SimulationErrorCode.COMMAND_ERROR: CommandError,
        SimulationErrorCode.HISTORY_ERROR: HistoryError,
        SimulationErrorCode.EXTENSION_ERROR: ExtensionError,
        SimulationErrorCode.STORAGE_ERROR: StorageError,
        SimulationErrorCode.IO_ERROR: IOError,
        SimulationErrorCode.LOGS_ERROR: LogsError,
        SimulationErrorCode.INITIALIZATION_ERROR: InitializationError,
        SimulationErrorCode.MANAGER_ERROR: ManagerError,
        SimulationErrorCode.PYTHON_ERROR: PythonError,
    }
    return error_classes.get(code, GeneralSimulationError)

class ConfigurationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CONFIGURATION, message, details)

class StorageInitializationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.STORAGE_INITIALIZATION, message, details)

class BackendInitializationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.BACKEND_INITIALIZATION, message, details)

class DatabaseInitializationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.DATABASE_INITIALIZATION, message, details)

class MeshSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MESH_SETUP, message, details)

class MassMatrixSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MASS_MATRIX_SETUP, message, details)

class ExternalForcesSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.EXTERNAL_FORCES_SETUP, message, details)

class HyperElasticSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.HYPERELASTIC_SETUP, message, details)

class CollisionSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.COLLISION_SETUP, message, details)

class BoundaryConditionsSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.BOUNDARY_CONDITIONS_SETUP, message, details)

class LineSearchSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.LINE_SEARCH_SETUP, message, details)

class LinearSolverSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.LINEAR_SOLVER_SETUP, message, details)

class OptimizerSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.OPTIMIZER_SETUP, message, details)

class ParametersSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.PARAMETERS_SETUP, message, details)

class PotentialSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.POTENTIAL_SETUP, message, details)

class GradientSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.GRADIENT_SETUP, message, details)

class HessianSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.HESSIAN_SETUP, message, details)

class BarrierUpdaterSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.BARRIER_UPDATER_SETUP, message, details)

class CCDSetupError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CCD_SETUP, message, details)

class SimulationLoopError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.SIMULATION_LOOP, message, details)

class CommandProcessingError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.COMMAND_PROCESSING, message, details)

class MemoryAllocationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MEMORY_ALLOCATION, message, details)

class FileIOError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.FILE_IO, message, details)

class NetworkCommunicationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.NETWORK_COMMUNICATION, message, details)

class ThreadingError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.THREADING, message, details)

class GPUInitializationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.GPU_INITIALIZATION, message, details)

class GPUComputationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.GPU_COMPUTATION, message, details)

class InputValidationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.INPUT_VALIDATION, message, details)

class ResourceAllocationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.RESOURCE_ALLOCATION, message, details)

class TimingConstraintError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.TIMING_CONSTRAINT, message, details)

class GeneralSimulationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.GENERAL, message, details)

class LineSearchError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.LINE_SEARCH, message, details)

class GradientCalculationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.GRADIENT_CALCULATION, message, details)

class BackendError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.BACKEND_ERROR, message, details)

class NetworkError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.NETWORK_ERROR, message, details)

class FileError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.FILE_ERROR, message, details)

class MemoryError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MEMORY_ERROR, message, details)

class CPUError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CPU_ERROR, message, details)

class BoardError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.BOARD_ERROR, message, details)

class ServerError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.SERVER_ERROR, message, details)

class ConnectionError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CONNECTION_ERROR, message, details)

class ConfigError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CONFIG_ERROR, message, details)

class CoreError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CORE_ERROR, message, details)

class MathError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MATH_ERROR, message, details)

class StateError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.STATE_ERROR, message, details)

class ControllerError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.CONTROLLER_ERROR, message, details)

class CommandError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.COMMAND_ERROR, message, details)

class HistoryError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.HISTORY_ERROR, message, details)

class ExtensionError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.EXTENSION_ERROR, message, details)

class StorageError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.STORAGE_ERROR, message, details)

class IOError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.IO_ERROR, message, details)

class LogsError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.LOGS_ERROR, message, details)

class InitializationError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.INITIALIZATION_ERROR, message, details)

class ManagerError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.MANAGER_ERROR, message, details)

class PythonError(SimulationError):
    def __init__(self, message: str, details: str = None):
        super().__init__(SimulationErrorCode.PYTHON_ERROR, message, details)
