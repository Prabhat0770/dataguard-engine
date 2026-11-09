class DataGuardError(Exception):
    """Base exception class for all DataGuard pipeline errors."""
    def __init__(self, message: str, step: str = "Unknown"):
        self.message = message
        self.step = step
        super().__init__(f"[{self.step}] Error: {self.message}")


class EmptyDatasetError(DataGuardError):
    """Raised when the loaded dataset has 0 rows or 0 columns."""
    pass


class InvalidFileFormatError(DataGuardError):
    """Raised when an unsupported file extension is passed."""
    pass