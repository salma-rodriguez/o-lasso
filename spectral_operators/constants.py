"""
spectral_operator.constants
===========================

Shared library-wide constants.
"""

from __future__ import annotations

from enum import Enum


# ===========================================================================
# Numerical Constants
# ===========================================================================

DEFAULT_TOL = 1e-10
DEFAULT_ZERO_TOL = 1e-12

DEFAULT_NORM = "fro"
DEFAULT_ORDERING = "abs"

DEFAULT_BOUNDARY_WIDTH = 1

DEFAULT_NORMALIZE = True


# ===========================================================================
# Plotting Defaults
# ===========================================================================

DEFAULT_FIGSIZE = (8, 6)
DEFAULT_DPI = 150


# ===========================================================================
# Metadata
# ===========================================================================

PACKAGE_NAME = "spectral_operators"

PACKAGE_VERSION = "0.1.0-dev"

AUTHOR = "Salma Y. Rodriguez"

LICENSE = "MIT"


# ===========================================================================
# Enumerations
# ===========================================================================

class Ordering(str, Enum):
    ABS = "abs"
    REAL = "real"
    IMAG = "imag"
    COMPLEX = "complex"


class WeightRule(str, Enum):
    UNIFORM = "uniform"
    INVERSE = "inverse"
    LOG = "log"
    LOG_INVERSE = "log_inverse"


class EvolutionType(str, Enum):
    UNITARY = "unitary"
    SEMIGROUP = "semigroup"
