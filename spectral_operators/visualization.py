"""
spectral_operators.visualization
================================

Backend-neutral visualization data helpers.

This module prepares spectral, matrix, geometric, diagnostic, and
zeta-correspondence data for plotting libraries such as Matplotlib,
Plotly, or notebook-specific renderers.

No plotting backend is required by this module.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import OperatorError
from .core.utilities import (
    readonly_array,
    require_nonnegative_integer,
    require_positive_integer,
)
from .diagnostics import OperatorDiagnostics
from .geometry import GeometryAnalyzer
from .spectrum import SpectralAnalyzer
from .zeta import ZetaCorrespondence, ZetaZeroSet


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _validate_operator(
    operator: LinearOperator,
) -> LinearOperator:
    """
    Validate and return a LinearOperator.
    """

    if not isinstance(operator, LinearOperator):
        raise OperatorError(
            "operator must be a LinearOperator."
        )

    return operator


# ===========================================================================
# Spectrum Visualization Data
# ===========================================================================

class SpectrumData:
    """
    Prepare eigenvalue data for visualization.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        ordering: str = "abs",
    ):
        operator = _validate_operator(operator)

        eigenvalues = SpectralAnalyzer(
            operator
        ).sorted_eigenvalues(ordering)

        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(
            self,
            "eigenvalues",
            readonly_array(
                eigenvalues,
                name="eigenvalues",
                ndim=1,
            ),
        )

    def as_dict(self) -> dict:
        """
        Return JSON-compatible spectrum data.
        """

        return {
            "operator": self.operator.name,
            "ordering": self.ordering,
            "real": self.eigenvalues.real.tolist(),
            "imag": self.eigenvalues.imag.tolist(),
            "magnitude": self.magnitudes().tolist(),
            "phase": self.phases().tolist(),
        }

    def complex_points(self) -> np.ndarray:
        """
        Return ``(real, imaginary)`` coordinate pairs.
        """

        points = np.column_stack((
            self.eigenvalues.real,
            self.eigenvalues.imag,
        ))

        return readonly_array(
            points,
            name="complex points",
            ndim=2,
        )

    def magnitudes(self) -> np.ndarray:
        """
        Return eigenvalue magnitudes.
        """

        return readonly_array(
            np.abs(self.eigenvalues),
            name="eigenvalue magnitudes",
            ndim=1,
        )

    def phases(self) -> np.ndarray:
        """
        Return eigenvalue phases in radians.
        """

        return readonly_array(
            np.angle(self.eigenvalues),
            name="eigenvalue phases",
            ndim=1,
        )


# ===========================================================================
# Matrix Visualization Data
# ===========================================================================

class MatrixData:
    """
    Prepare operator matrix data for visualization.
    """

    def __init__(
        self,
        operator: LinearOperator,
    ):
        operator = _validate_operator(operator)

        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "matrix",
            operator.matrix,
        )

    def as_dict(self) -> dict:
        """
        Return JSON-compatible matrix data.
        """

        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "real": self.real().tolist(),
            "imag": self.imag().tolist(),
            "magnitude": self.magnitude().tolist(),
            "phase": self.phase().tolist(),
        }

    def imag(self) -> np.ndarray:
        """
        Return the imaginary part.
        """

        return readonly_array(
            self.matrix.imag,
            name="imaginary matrix",
            ndim=2,
        )

    def magnitude(self) -> np.ndarray:
        """
        Return entrywise absolute values.
        """

        return readonly_array(
            np.abs(self.matrix),
            name="matrix magnitude",
            ndim=2,
        )

    def phase(self) -> np.ndarray:
        """
        Return entrywise phases in radians.
        """

        return readonly_array(
            np.angle(self.matrix),
            name="matrix phase",
            ndim=2,
        )

    def real(self) -> np.ndarray:
        """
        Return the real part.
        """

        return readonly_array(
            self.matrix.real,
            name="real matrix",
            ndim=2,
        )


# ===========================================================================
# Geometry Visualization Data
# ===========================================================================

class GeometryData:
    """
    Prepare geometry diagnostic data for visualization.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        boundary_width: int = 1,
    ):
        operator = _validate_operator(operator)
        boundary_width = require_positive_integer(
            boundary_width,
            name="boundary_width",
        )

        geometry = GeometryAnalyzer(
            operator,
            boundary_width=boundary_width,
        )

        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "boundary_width",
            boundary_width,
        )
        object.__setattr__(self, "geometry", geometry)

    def as_dict(
        self,
        *,
        bandwidth: int = 1,
    ) -> dict:
        """
        Return JSON-compatible geometry data.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )

        return {
            "operator": self.operator.name,
            "boundary_width": self.boundary_width,
            "defects": self.defect_bars(),
            "ratios": self.ratio_bars(
                bandwidth=bandwidth
            ),
        }

    def defect_bars(self) -> dict:
        """
        Return defect values suitable for bar plots.
        """

        return self.geometry.defects()

    def ratio_bars(
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

        return self.geometry.ratios(
            bandwidth=bandwidth
        )


# ===========================================================================
# Zeta Visualization Data
# ===========================================================================

class ZetaData:
    """
    Prepare zeta-correspondence data for visualization.
    """

    def __init__(
        self,
        correspondence: ZetaCorrespondence,
    ):
        if not isinstance(
            correspondence,
            ZetaCorrespondence,
        ):
            raise OperatorError(
                "correspondence must be a ZetaCorrespondence."
            )

        object.__setattr__(
            self,
            "correspondence",
            correspondence,
        )

    def as_dict(
        self,
        n: int | None = None,
    ) -> dict:
        """
        Return JSON-compatible correspondence data.
        """

        spectral_values, zero_ordinates = (
            self.correspondence.paired_values(
                n=n
            )
        )

        errors = (
            spectral_values
            - zero_ordinates
        )

        return {
            "count": int(
                len(spectral_values)
            ),
            "spectral_values":
                spectral_values.tolist(),
            "zero_ordinates":
                zero_ordinates.tolist(),
            "errors":
                errors.tolist(),
        }

    def error_series(
        self,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Return spectral-minus-zeta ordinate errors.
        """

        spectral_values, zero_ordinates = (
            self.correspondence.paired_values(
                n=n
            )
        )

        return readonly_array(
            spectral_values - zero_ordinates,
            name="error series",
            ndim=1,
        )

    def paired_points(
        self,
        n: int | None = None,
    ) -> np.ndarray:
        """
        Return ``(spectral ordinate, zero ordinate)`` pairs.
        """

        spectral_values, zero_ordinates = (
            self.correspondence.paired_values(
                n=n
            )
        )

        points = np.column_stack((
            spectral_values,
            zero_ordinates,
        ))

        return readonly_array(
            points,
            name="zeta paired points",
            ndim=2,
        )


# ===========================================================================
# Visualization Bundle
# ===========================================================================

class VisualizationBundle:
    """
    High-level visualization-data bundle for a LinearOperator.
    """

    def __init__(
        self,
        operator: LinearOperator,
    ):
        object.__setattr__(
            self,
            "operator",
            _validate_operator(operator),
        )

    def as_dict(
        self,
        *,
        ordering: str = "abs",
        bandwidth: int = 1,
        boundary_width: int = 1,
    ) -> dict:
        """
        Return all standard visualization data.
        """

        bandwidth = require_nonnegative_integer(
            bandwidth,
            name="bandwidth",
        )
        boundary_width = require_positive_integer(
            boundary_width,
            name="boundary_width",
        )

        return {
            "operator": self.operator.name,
            "spectrum":
                self.spectrum(
                    ordering=ordering
                ).as_dict(),
            "matrix":
                self.matrix().as_dict(),
            "geometry":
                self.geometry(
                    boundary_width=boundary_width
                ).as_dict(
                    bandwidth=bandwidth
                ),
            "diagnostics":
                self.diagnostics(),
        }

    def diagnostics(self) -> dict:
        """
        Return operator diagnostics.
        """

        return OperatorDiagnostics(
            self.operator
        ).summary()

    def geometry(
        self,
        *,
        boundary_width: int = 1,
    ) -> GeometryData:
        """
        Return geometry visualization data.
        """

        return GeometryData(
            self.operator,
            boundary_width=boundary_width,
        )

    def matrix(self) -> MatrixData:
        """
        Return matrix visualization data.
        """

        return MatrixData(
            self.operator
        )

    def spectrum(
        self,
        *,
        ordering: str = "abs",
    ) -> SpectrumData:
        """
        Return spectrum visualization data.
        """

        return SpectrumData(
            self.operator,
            ordering=ordering,
        )
