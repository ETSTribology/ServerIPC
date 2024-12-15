# simulation/errors.py

class CommandError(Exception):
    """Base class for command-related errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CommandFailedError(CommandError):
    """Exception raised when a command fails to execute."""
    pass


class CannotConnectError(CommandError):
    """Exception raised when a connection error occurs."""
    pass


class InvalidParametersError(CommandError):
    """Exception raised when invalid parameters are provided."""
    pass
