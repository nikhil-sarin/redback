"""Exceptions with explicit inference semantics."""


class ModelDomainError(ValueError):
    """Parameters are outside a model's physical domain, not a solver failure."""
