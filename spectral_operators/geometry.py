"""
spectral_operators.geometry
====================+++++++

Geometric diagnostics for LinearOperator objects.

This module provides symmetry, boundary, locality, and combined
operator-geometry diagnostics.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import OperatorError
from .core.utilities import (
    readonly_array,
    require_nonnegative_integer,
    require_positive_integer,
    require_probability,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================


def _validate_operator(operator: LinearOperator) -> LinearOperator:
    """
    Validate and return a LinearOperator.
    """

    if not isinstance(operator, LinearOperator):
        raise OperatorError(
            "operator must be a LinearOperator."
        )

    return operator


def _relative_defect(
    defect: float,
    operator: LinearOperator,
) -> float:
    """
    Normalize a defect by the operator's Frobenius norm.
    """

    denominator = operator.norm("fro")

    if denominator == 0:
        return 0.0

    return float(defect / denominator)


# ===========================================================================
# Symmetry Diagnostics
# ===========================================================================

class SymmetryAnalyzer:
    """
    Analyze symmetric, skew-symmetric, Hermitian, and anti-Hermitian
    structure.
    """

    def __init__(self, operator: LinearOperator):
        object.__setattr__(
            self,
            "operator",
            _validate_operator(operator),
        )

    def antihermitian_defect(
        self,
        *,
        relative: bool = False,
    ) -> float:
        """
        Measure the defect from anti-Hermitian structure.

        Computes

            ||A + A†||_F.
        """

        matrix = self.operator.matrix
        defect = float(
            np.linalg.norm(
                matrix + matrix.conj().T,
                ord="fro",
            )
        )

        if relative:
            return _relative_defect(
                defect,
                self.operator,
            )

        return defect

    def hermitian_defect(
        self,
        *,
        relative: bool = False,
    ) -> float:
        """
        Measure the defect from Hermitian structure.

        Computes

            ||A - A†||_F.
        """

        matrix = self.operator.matrix
        defect = float(
            np.linalg.norm(
                matrix - matrix.conj().T,
                ord="fro",
            )
        )

        if relative:
            return _relative_defect(
                defect,
                self.operator,
            )

        return defect

    def skew_defect(
        self,
        *,
        relative: bool = False,
    ) -> float:
        """
        Measure the defect from skew-symmetry.

        Computes

            ||A + A^T||_F.
        """

        matrix = self.operator.matrix
        defect = float(
            np.linalg.norm(
                matrix + matrix.T,
                ord="fro",
            )
        )

        if relative:
            return _relative_defect(
                defect,
                self.operator,
            )

        return defect

    def summary(self) -> dict:
        """
        Return symmetry diagnostic results.
        """

        return {
            "symmetric": self.operator.is_symmetric(),
            "skew": self.operator.is_skew(),
            "hermitian": self.operator.is_hermitian(),
            "antihermitian": self.operator.is_antihermitian(),
            "symmetric_defect": self.symmetric_defect(),
            "skew_defect": self.skew_defect(),
            "hermitian_defect": self.hermitian_defect(),
            "antihermitian_defect": self.antihermitian_defect(),
            "relative_symmetric_defect":
                self.symmetric_defect(relative=True),
            "relative_skew_defect":
                self.skew_defect(relative=True),
            "relative_hermitian_defect":
                self.hermitian_defect(relative=True),
            "relative_antihermitian_defect":
                self.antihermitian_defect(relative=True),
        }

    def symmetric_defect(
        self,
        *,
        relative: bool = False,
    ) -> float:
        """
        Measure the defect from symmetry.

        Computes

            ||A - A^T||_F.
        """

        matrix = self.operator.matrix
        defect = float(
            np.linalg.norm(
                matrix - matrix.T,
                ord="fro",
            )
        )

        if relative:
            return _relative_defect(
                defect,
                self.operator,
            )

        return defect


# ===========================================================================
# Boundary Diagnostics
# ===========================================================================

class BoundaryAnalyzer:
    """
    Analyze boundary-localized operator mass.

    The boundary consists of the first and last ``width`` rows and
    columns of the matrix representation.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        width: int = 1,
    ):
        operator = _validate_operator(operator)
        width = require_positive_integer(
            width,
            name="width",
        )

        if 2 * width > min(operator.shape):
            raise OperatorError(
                "width is too large for the operator shape."
            )

        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "width", width)

    def boundary_mask(self) -> np.ndarray:
        """
        Return a read-only boolean boundary mask.
        """

        rows, cols = self.operator.shape
        width = self.width

        mask = np.zeros(
            (rows, cols),
            dtype=bool,
        )

        mask[:width, :] = True
        mask[-width:, :] = True
        mask[:, :width] = True
        mask[:, -width:] = True

        return readonly_array(
            mask,
            name="boundary mask",
            ndim=2,
        )

    def boundary_norm(self) -> float:
        """
        Return the Frobenius norm of boundary entries.
        """

        return float(
            np.linalg.norm(
                self.operator.matrix[
                    self.boundary_mask()
                ]
            )
        )

    def boundary_ratio(self) -> float:
        """
        Return boundary norm divided by total norm.
        """

        total = self.total_norm()

        if total == 0:
            return 0.0

        return float(
            self.boundary_norm() / total
        )

    def interior_mask(self) -> np.ndarray:
        """
        Return a read-only boolean interior mask.
        """

        return readonly_array(
            ~self.boundary_mask(),
            name="interior mask",
            ndim=2,
        )

    def interior_norm(self) -> float:
        """
        Return the Frobenius norm of interior entries.
        """

        return float(
            np.linalg.norm(
                self.operator.matrix[
                    self.interior_mask()
                ]
            )
        )

    def interior_ratio(self) -> float:
        """
        Return interior norm divided by total norm.
        """

        total = self.total_norm()

        if total == 0:
            return 0.0

        return float(
            self.interior_norm() / total
        )

    def summary(self) -> dict:
        """
        Return boundary and interior diagnostics.
        """

        return {
            "width": self.width,
            "boundary_norm": self.boundary_norm(),
            "interior_norm": self.interior_norm(),
            "total_norm": self.total_norm(),
            "boundary_ratio": self.boundary_ratio(),
            "interior_ratio": self.interior_ratio(),
        }

    def total_norm(self) -> float:
        """
        Return the full Frobenius norm.
        """

        return self.operator.norm("fro")


# ===========================================================================
# Locality Diagnostics
# ===========================================================================

class LocalityAnalyzer:
    """
    Analyze matrix locality relative to the main diagonal.
    """

    def __init__(self, operator: LinearOperator):
        object.__setattr__(
            self,
            "operator",
            _validate_operator(operator),
        )

    def band_mask(
        self,
        bandwidth: int,
    ) -> np.ndarray:
        """
        Return a read-only mask selecting ``|i - j| <= bandwidth``.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )

        mask = (
            self.distance_matrix()
            <= bandwidth
        )

        return readonly_array(
            mask,
            name="band mask",
            ndim=2,
        )

    def band_norm(
        self,
        bandwidth: int,
    ) -> float:
        """
        Return the Frobenius norm inside a diagonal band.
        """

        return float(
            np.linalg.norm(
                self.operator.matrix[
                    self.band_mask(bandwidth)
                ]
            )
        )

    def distance_matrix(self) -> np.ndarray:
        """
        Return the read-only matrix ``|i - j|``.
        """

        rows, cols = self.operator.shape

        row_indices = np.arange(rows)[:, None]
        column_indices = np.arange(cols)[None, :]

        distances = np.abs(
            row_indices - column_indices
        )

        return readonly_array(
            distances,
            name="distance matrix",
            ndim=2,
        )

    def effective_bandwidth(
        self,
        threshold: float = 0.95,
    ) -> int:
        """
        Return the smallest bandwidth capturing the requested norm ratio.
        """

        threshold = require_probability(
            threshold,
            name="threshold",
            inclusive=True,
        )

        max_bandwidth = max(
            self.operator.shape
        ) - 1

        for bandwidth in range(
            max_bandwidth + 1
        ):
            if (
                self.locality_ratio(bandwidth)
                >= threshold
            ):
                return bandwidth

        return max_bandwidth

    def locality_ratio(
        self,
        bandwidth: int,
    ) -> float:
        """
        Return in-band norm divided by total norm.
        """

        total = self.operator.norm("fro")

        if total == 0:
            return 0.0

        return float(
            self.band_norm(bandwidth)
            / total
        )

    def off_band_norm(
        self,
        bandwidth: int,
    ) -> float:
        """
        Return the Frobenius norm outside a diagonal band.
        """

        mask = ~self.band_mask(bandwidth)

        return float(
            np.linalg.norm(
                self.operator.matrix[mask]
            )
        )

    def off_locality_ratio(
        self,
        bandwidth: int,
    ) -> float:
        """
        Return off-band norm divided by total norm.
        """

        total = self.operator.norm("fro")

        if total == 0:
            return 0.0

        return float(
            self.off_band_norm(bandwidth)
            / total
        )

    def summary(
        self,
        bandwidth: int = 1,
    ) -> dict:
        """
        Return locality diagnostics.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )

        return {
            "bandwidth": bandwidth,
            "band_norm": self.band_norm(
                bandwidth
            ),
            "off_band_norm": self.off_band_norm(
                bandwidth
            ),
            "locality_ratio": self.locality_ratio(
                bandwidth
            ),
            "off_locality_ratio":
                self.off_locality_ratio(
                    bandwidth
                ),
            "effective_bandwidth_95":
                self.effective_bandwidth(
                    0.95
                ),
        }


# ===========================================================================
# Geometry Diagnostics
# ===========================================================================

class GeometryAnalyzer:
    """
    High-level interface combining symmetry, boundary, and locality
    diagnostics.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        boundary_width: int = 1,
    ):
        operator = _validate_operator(operator)

        symmetry = SymmetryAnalyzer(operator)
        boundary = BoundaryAnalyzer(
            operator,
            width=boundary_width,
        )
        locality = LocalityAnalyzer(operator)

        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "boundary_width",
            boundary.width,
        )
        object.__setattr__(self, "symmetry", symmetry)
        object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "locality", locality)

    def defects(self) -> dict:
        """
        Return principal defect diagnostics.
        """

        return {
            "symmetric_defect":
                self.symmetry.symmetric_defect(),
            "skew_defect":
                self.symmetry.skew_defect(),
            "hermitian_defect":
                self.symmetry.hermitian_defect(),
            "antihermitian_defect":
                self.symmetry.antihermitian_defect(),
            "boundary_ratio":
                self.boundary.boundary_ratio(),
        }

    def ratios(
        self,
        *,
        bandwidth: int = 1,
    ) -> dict:
        """
        Return normalized geometry ratios.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )

        return {
            "relative_symmetric_defect":
                self.symmetry.symmetric_defect(
                    relative=True
                ),
            "relative_skew_defect":
                self.symmetry.skew_defect(
                    relative=True
                ),
            "relative_hermitian_defect":
                self.symmetry.hermitian_defect(
                    relative=True
                ),
            "relative_antihermitian_defect":
                self.symmetry.antihermitian_defect(
                    relative=True
                ),
            "boundary_ratio":
                self.boundary.boundary_ratio(),
            "interior_ratio":
                self.boundary.interior_ratio(),
            "locality_ratio":
                self.locality.locality_ratio(
                    bandwidth
                ),
            "off_locality_ratio":
                self.locality.off_locality_ratio(
                    bandwidth
                ),
        }

    def summary(
        self,
        *,
        bandwidth: int = 1,
    ) -> dict:
        """
        Return combined geometry diagnostics.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )

        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "field": self.operator.field.value,
            "symmetry": self.symmetry.summary(),
            "boundary": self.boundary.summary(),
            "locality": self.locality.summary(
                bandwidth=bandwidth
            ),
        }
