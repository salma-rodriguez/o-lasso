"""
spectral_operators.constants
============================

Shared constants, defaults, enumerations, and package metadata for the
spectral_operators subpackage.
"""

from __future__ import annotations

from enum import Enum


# ===========================================================================
# Numerical Defaults
# ===========================================================================

DEFAULT_TOL: float = 1e-10
DEFAULT_ZERO_TOL: float = 1e-12

DEFAULT_NORM: str = "fro"
DEFAULT_ORDERING: str = "abs"

DEFAULT_BOUNDARY_WIDTH: int = 1
DEFAULT_NORMALIZE: bool = True


# ===========================================================================
# Visualization Defaults
# ===========================================================================

DEFAULT_FIGSIZE: tuple[int, int] = (8, 6)
DEFAULT_DPI: int = 150


# ===========================================================================
# Project Metadata
# ===========================================================================

PROJECT_NAME: str = "operatorlab"
PACKAGE_NAME: str = "spectral_operators"

PACKAGE_VERSION: str = "0.1.0-dev"

AUTHOR: str = "Salma Y. Rodriguez"
LICENSE: str = "MIT"


# ===========================================================================
# Enumerations
# ===========================================================================

class Ordering(str, Enum):
    """
    Supported spectral ordering conventions.
    """

    ABS = "abs"
    REAL = "real"
    IMAG = "imag"
    COMPLEX = "complex"


class WeightRule(str, Enum):
    """
    Supported scalar weight-generation rules.
    """

    UNIFORM = "uniform"
    INVERSE = "inverse"
    LOG = "log"
    LOG_INVERSE = "log_inverse"


class EvolutionType(str, Enum):
    """
    Supported evolution families.
    """

    UNITARY = "unitary"
    SEMIGROUP = "semigroup"


__all__ = [
    "AUTHOR",
    "DEFAULT_BOUNDARY_WIDTH",
    "DEFAULT_DPI",
    "DEFAULT_FIGSIZE",
    "DEFAULT_NORMALIZE",
    "DEFAULT_NORM",
    "DEFAULT_ORDERING",
    "DEFAULT_TOL",
    "DEFAULT_ZERO_TOL",
    "EvolutionType",
    "LICENSE",
    "Ordering",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "PROJECT_NAME",
    "WeightRule",
]
