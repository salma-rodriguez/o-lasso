"""
spectral_operator.spectrum
==========================

Spectral analysis tools for matrix-backed LinearOperator objects.

This module contains reusable classes and functions for studying
eigenvalues, eigenvectors, singular values, resolvents, spectral
partitions, resolvent diagnostics, spacing statistics and spectral diagnostics.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import (
    NonSquareOperatorError,
    OperatorError,
    SingularOperatorError,
)
from .core.utilities import (
    as_one_dimensional_array,
    readonly_array,
    require_nonnegative_integer,
    require_probability,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================

_SUPPORTED_ORDERINGS = {
    "abs",
    "real",
    "imag",
    "complex",
}


def _sorted_values(
    values,
    ordering: str = "abs",
) -> np.ndarray:
    """
    Return a sorted copy of a one-dimensional spectral array.
    """

    array = as_one_dimensional_array(
        values,
        name="eigenvalues",
        copy=True,
        allow_empty=True,
    )

    if ordering not in _SUPPORTED_ORDERINGS:
        raise OperatorError(
            "ordering must be one of: "
            "'abs', 'real', 'imag', or 'complex'."
        )

    if ordering == "abs":
        indices = np.argsort(np.abs(array))

    elif ordering == "real":
        indices = np.argsort(array.real)

    elif ordering == "imag":
        indices = np.argsort(array.imag)

    else:
        indices = np.lexsort(
            (array.imag, array.real)
        )

    return array[indices]


# ===========================================================================
# Spectral Analyzer
# ===========================================================================

class SpectralAnalyzer:
    """
    High-level spectral analysis wrapper for LinearOperator objects.
    """

    def __init__(self, operator: LinearOperator):
        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        object.__setattr__(self, "operator", operator)

    def eigendecomposition(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return eigenvalues and right eigenvectors.
        """

        return self.operator.eigendecomposition()

    def eigenvalues(self) -> np.ndarray:
        """
        Return operator eigenvalues.
        """

        return self.operator.eigenvalues()

    def eigenvectors(self) -> np.ndarray:
        """
        Return right eigenvectors.
        """

        return self.operator.eigenvectors()

    def partition(
        self,
        *,
        alpha: float = 0.13,
        ordering: str = "abs",
    ) -> "SpectralPartition":
        """
        Partition the spectrum into low, bulk, and high regions.
        """

        return SpectralPartition(
            self.eigenvalues(),
            alpha=alpha,
            ordering=ordering,
        )

    def resolvent(self) -> "ResolventAnalyzer":
        """
        Return a resolvent analyzer for the operator.
        """

        return ResolventAnalyzer(self.operator)

    def sorted_eigenvalues(
        self,
        ordering: str = "abs",
    ) -> np.ndarray:
        """
        Return eigenvalues sorted according to the selected ordering.
        """

        return _sorted_values(
            self.eigenvalues(),
            ordering=ordering,
        )

    def statistics(
        self,
        *,
        ordering: str = "real",
    ) -> "SpectralStatistics":
        """
        Return spectral spacing statistics.
        """

        return SpectralStatistics(
            self.eigenvalues(),
            ordering=ordering,
        )

    def svd(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the reduced singular value decomposition.
        """

        return self.operator.svd()


# ===========================================================================
# Spectral Partition
# ===========================================================================

class SpectralPartition:
    """
    Partition eigenvalues into low, bulk, and high spectral regions.

    Parameters
    ----------
    eigenvalues
        One-dimensional array of spectral values.

    alpha
        Fraction assigned to each edge partition. Must satisfy
        ``0 <= alpha < 0.5``.

    ordering
        Sorting convention.
    """

    def __init__(
        self,
        eigenvalues,
        *,
        alpha: float = 0.13,
        ordering: str = "abs",
    ):
        values = as_one_dimensional_array(
            eigenvalues,
            name="eigenvalues",
            copy=True,
            allow_empty=True,
        )

        alpha = require_probability(
            alpha,
            name="alpha",
            inclusive=True,
        )

        if alpha >= 0.5:
            raise OperatorError(
                "alpha must satisfy 0 <= alpha < 0.5."
            )

        sorted_values = _sorted_values(
            values,
            ordering=ordering,
        )

        edge_size = int(
            alpha * len(sorted_values)
        )

        if edge_size == 0:
            low = np.array(
                [],
                dtype=sorted_values.dtype,
            )
            bulk = sorted_values.copy()
            high = np.array(
                [],
                dtype=sorted_values.dtype,
            )
        else:
            low = sorted_values[:edge_size]
            bulk = sorted_values[
                edge_size:-edge_size
            ]
            high = sorted_values[-edge_size:]

        object.__setattr__(
            self,
            "eigenvalues",
            readonly_array(
                values,
                name="eigenvalues",
                ndim=1,
            ),
        )
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(
            self,
            "sorted_values",
            readonly_array(
                sorted_values,
                name="sorted values",
                ndim=1,
            ),
        )
        object.__setattr__(self, "k", edge_size)
        object.__setattr__(
            self,
            "low",
            readonly_array(
                low,
                name="low spectrum",
                ndim=1,
            ),
        )
        object.__setattr__(
            self,
            "bulk",
            readonly_array(
                bulk,
                name="bulk spectrum",
                ndim=1,
            ),
        )
        object.__setattr__(
            self,
            "high",
            readonly_array(
                high,
                name="high spectrum",
                ndim=1,
            ),
        )

    def as_tuple(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return mutable copies of the three partitions.
        """

        return (
            self.low.copy(),
            self.bulk.copy(),
            self.high.copy(),
        )

    def sizes(self) -> tuple[int, int, int]:
        """
        Return low, bulk, and high partition sizes.
        """

        return (
            len(self.low),
            len(self.bulk),
            len(self.high),
        )


# ===========================================================================
# Resolvent Analysis
# ===========================================================================

class ResolventAnalyzer:
    """
    Resolvent-based spectral diagnostics.

    The resolvent is

        R(z) = (A - zI)^(-1).
    """

    def __init__(self, operator: LinearOperator):
        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        if not operator.is_square:
            raise NonSquareOperatorError(
                "Resolvent analysis requires a square operator."
            )

        object.__setattr__(self, "operator", operator)

    def determinant(self, z):
        """
        Return det(A - zI).
        """

        shifted = self._shifted_matrix(z)

        return np.linalg.det(shifted)

    def matrix(self, z) -> np.ndarray:
        """
        Return the resolvent matrix.
        """

        shifted = self._shifted_matrix(z)

        try:
            return np.linalg.inv(shifted)

        except np.linalg.LinAlgError as exc:
            raise SingularOperatorError(
                f"Resolvent is singular at z={z!r}."
            ) from exc

    def norm(
        self,
        z,
        kind="spectral",
    ) -> float:
        """
        Return the requested norm of R(z).
        """

        resolvent = LinearOperator(
            self.matrix(z),
            name=f"R({z})",
            metadata={
                "operator": self.operator.name,
                "spectral_parameter": z,
            },
        )

        return resolvent.norm(kind)

    def trace(self, z):
        """
        Return tr(R(z)).
        """

        return np.trace(
            self.matrix(z)
        )

    def _shifted_matrix(self, z) -> np.ndarray:
        """
        Return A - zI.
        """

        matrix = self.operator.matrix
        identity = np.eye(
            self.operator.rows,
            dtype=np.result_type(matrix, z),
        )

        return matrix - z * identity


# ===========================================================================
# Spectral Statistics
# ===========================================================================

class SpectralStatistics:
    """
    Consecutive spectral-gap and spacing statistics.
    """

    def __init__(
        self,
        eigenvalues,
        *,
        ordering: str = "real",
    ):
        values = as_one_dimensional_array(
            eigenvalues,
            name="eigenvalues",
            copy=True,
            allow_empty=True,
        )

        sorted_values = _sorted_values(
            values,
            ordering=ordering,
        )

        object.__setattr__(
            self,
            "eigenvalues",
            readonly_array(
                values,
                name="eigenvalues",
                ndim=1,
            ),
        )
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(
            self,
            "sorted_values",
            readonly_array(
                sorted_values,
                name="sorted values",
                ndim=1,
            ),
        )

    def gaps(self) -> np.ndarray:
        """
        Return consecutive ordered spectral differences.
        """

        return np.diff(
            self.sorted_values
        )

    def max_spacing(self) -> float:
        """
        Return the maximum absolute spacing.
        """

        spacings = self.spacings()

        if spacings.size == 0:
            return 0.0

        return float(
            np.max(spacings)
        )

    def mean_spacing(self) -> float:
        """
        Return the mean absolute spacing.
        """

        spacings = self.spacings()

        if spacings.size == 0:
            return 0.0

        return float(
            np.mean(spacings)
        )

    def min_spacing(self) -> float:
        """
        Return the minimum absolute spacing.
        """

        spacings = self.spacings()

        if spacings.size == 0:
            return 0.0

        return float(
            np.min(spacings)
        )

    def normalized_spacings(self) -> np.ndarray:
        """
        Return spacings divided by their mean.
        """

        spacings = self.spacings()

        if spacings.size == 0:
            return spacings

        mean = np.mean(spacings)

        if mean == 0:
            return spacings

        return spacings / mean

    def spacings(self) -> np.ndarray:
        """
        Return absolute consecutive spectral differences.
        """

        return np.abs(
            self.gaps()
        )

    def summary(self) -> dict:
        """
        Return basic spacing statistics.
        """

        return {
            "count": int(
                len(self.eigenvalues)
            ),
            "num_spacings": int(
                max(
                    len(self.eigenvalues) - 1,
                    0,
                )
            ),
            "ordering": self.ordering,
            "mean_spacing": self.mean_spacing(),
            "min_spacing": self.min_spacing(),
            "max_spacing": self.max_spacing(),
            "variance_spacing": self.variance_spacing(),
        }

    def variance_spacing(self) -> float:
        """
        Return the variance of absolute spacings.
        """

        spacings = self.spacings()

        if spacings.size == 0:
            return 0.0

        return float(
            np.var(spacings)
        )
