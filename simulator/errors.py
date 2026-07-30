"""Simulator-specific exception types."""


class SimulatorError(RuntimeError):
    """Base class for simulator errors."""


class SimulatorRuleError(SimulatorError):
    """Raised when an input violates a fail-fast rule assumption."""

