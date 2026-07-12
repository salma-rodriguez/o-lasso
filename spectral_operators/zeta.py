"""
spectral_operators.zeta
=======================

Zeta-oriented constructions and diagnostics.

This module connects operator spectra with spectral zeta functions,
zeta-zero data, correspondence diagnostics, and finite-dimensional
Hilbert--Pólya-style investigations.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import OperatorError
from .core.utilities import (
    as_one_dimensional_array,
    readonly_array,
    require_nonnegative_integer,
    require_positive_real,
)
from .spectrum import SpectralAnalyzer


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


def _spectral_ordinates(
    values,
) -> np.ndarray:
    """
    Convert spectral values into nonnegative comparison ordinates.

    Complex spectra are represented by absolute imaginary parts.
    Real spectra are represented by absolute values.
    """

    array = as_one_dimensional_array(
        values,
        name="spectral values",
        copy=True,
        allow_empty=True,
    )

    if np.iscomplexobj(array):
        ordinates = np.abs(array.imag)
    else:
        ordinates = np.abs(array)

    return readonly_array(
        ordinates,
        name="spectral ordinates",
        ndim=1,
    )


# ===========================================================================
# Spectral Zeta
# ===========================================================================

class SpectralZeta:
    """
    Spectral zeta function associated with operator eigenvalues.

    Computes

        zeta_A(s) = sum_n lambda_n^(-s)

    over the retained eigenvalues.

    Parameters
    ----------
    operator
        Square matrix-backed operator.

    discard_zeros
        If true, eigenvalues whose absolute value is at most
        ``zero_tol`` are removed.

    zero_tol
        Positive numerical threshold used for zero detection.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        discard_zeros: bool = True,
        zero_tol: float = 1e-12,
    ):
        operator = _validate_operator(operator)
        zero_tol = require_positive_real(
            zero_tol,
            name="zero_tol",
        )

        eigenvalues = SpectralAnalyzer(
            operator
        ).eigenvalues()

        if discard_zeros:
            eigenvalues = eigenvalues[
                np.abs(eigenvalues) > zero_tol
            ]

        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "discard_zeros",
            bool(discard_zeros),
        )
        object.__setattr__(self, "zero_tol", zero_tol)
        object.__setattr__(
            self,
            "eigenvalues",
            readonly_array(
                eigenvalues,
                name="eigenvalues",
                ndim=1,
            ),
        )

    def evaluate(self, s):
        """
        Evaluate the spectral zeta function at ``s``.
        """

        if self.eigenvalues.size == 0:
            return 0.0

        return np.sum(
            self.eigenvalues ** (-s)
        )

    def summary(self) -> dict:
        """
        Return spectral-zeta metadata.
        """

        return {
            "operator": self.operator.name,
            "num_eigenvalues": int(
                len(self.eigenvalues)
            ),
            "discard_zeros": self.discard_zeros,
            "zero_tol": self.zero_tol,
        }

    def values(
        self,
        s_values,
    ) -> np.ndarray:
        """
        Evaluate the spectral zeta function at multiple values of ``s``.
        """

        parameters = as_one_dimensional_array(
            s_values,
            name="s_values",
            copy=True,
            allow_empty=True,
        )

        return np.asarray([
            self.evaluate(s)
            for s in parameters
        ])


# ===========================================================================
# Zeta Zero Data
# ===========================================================================

class ZetaZeroSet:
    """
    Container for zeta-zero ordinates gamma_n.

    The corresponding critical-line points are represented as

        rho_n = 1/2 + i gamma_n.

    Notes
    -----
    This class stores supplied ordinates. It does not verify that the
    values are actual nontrivial zeros of the Riemann zeta function.
    """

    def __init__(
        self,
        gammas,
    ):
        ordinates = as_one_dimensional_array(
            gammas,
            name="gammas",
            dtype=float,
            copy=True,
            allow_empty=True,
        )

        if np.any(~np.isfinite(ordinates)):
            raise OperatorError(
                "gammas must contain only finite values."
            )

        if np.any(ordinates < 0):
            raise OperatorError(
                "gammas must be nonnegative."
            )

        ordinates = np.sort(ordinates)

        object.__setattr__(
            self,
            "gammas",
            readonly_array(
                ordinates,
                name="gammas",
                ndim=1,
            ),
        )

    def first(
        self,
        n: int,
    ) -> np.ndarray:
        """
        Return a copy of the first ``n`` ordinates.
        """

        n = require_nonnegative_integer(
            n,
            name="n",
        )

        return self.gammas[:n].copy()

    def spacings(self) -> np.ndarray:
        """
        Return consecutive ordinate spacings.
        """

        return np.diff(self.gammas)

    def summary(self) -> dict:
        """
        Return zero-set summary information.
        """

        if self.gammas.size == 0:
            return {
                "count": 0,
                "min_gamma": None,
                "max_gamma": None,
                "mean_spacing": None,
            }

        spacings = self.spacings()

        return {
            "count": int(len(self.gammas)),
            "min_gamma": float(
                np.min(self.gammas)
            ),
            "max_gamma": float(
                np.max(self.gammas)
            ),
            "mean_spacing": (
                float(np.mean(spacings))
                if spacings.size
                else None
            ),
        }

    @property
    def zeros(self) -> np.ndarray:
        """
        Return copies of the points ``1/2 + i gamma_n``.
        """

        return 0.5 + 1j * self.gammas


# ===========================================================================
# Zeta Correspondence
# ===========================================================================

class ZetaCorrespondence:
    """
    Compare an operator spectrum with supplied zeta-zero ordinates.
    """

    def __init__(
        self,
        operator: LinearOperator,
        zeros: ZetaZeroSet,
        *,
        ordering: str = "abs",
    ):
        operator = _validate_operator(operator)

        if not isinstance(zeros, ZetaZeroSet):
            raise OperatorError(
                "zeros must be a ZetaZeroSet."
            )

        spectrum = SpectralAnalyzer(
            operator
        ).sorted_eigenvalues(ordering)

        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "zeros", zeros)
        object.__setattr__(self, "ordering", ordering)
        object.__setattr__(
            self,
            "spectrum",
            readonly_array(
                spectrum,
                name="spectrum",
                ndim=1,
            ),
        )

    def compare(
        self,
        n: int | None = None,
    ) -> dict:
        """
        Compare paired spectral and zeta ordinates.
        """

        spectral_values, zero_values = (
            self.paired_values(n=n)
        )

        if spectral_values.size == 0:
            return {
                "count": 0,
                "mean_abs_error": None,
                "max_abs_error": None,
                "rms_error": None,
            }

        error = spectral_values - zero_values

        return {
            "count": int(
                len(spectral_values)
            ),
            "mean_abs_error": float(
                np.mean(np.abs(error))
            ),
            "max_abs_error": float(
                np.max(np.abs(error))
            ),
            "rms_error": float(
                np.sqrt(
                    np.mean(np.abs(error) ** 2)
                )
            ),
        }

    def paired_values(
        self,
        n: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return paired spectral ordinates and zeta-zero ordinates.
        """

        spectral_values = _spectral_ordinates(
            self.spectrum
        )
        zero_values = self.zeros.gammas

        count = min(
            len(spectral_values),
            len(zero_values),
        )

        if n is not None:
            n = require_nonnegative_integer(
                n,
                name="n",
            )
            count = min(count, n)

        return (
            spectral_values[:count].copy(),
            zero_values[:count].copy(),
        )


# ===========================================================================
# Hilbert--Polya Diagnostics
# ===========================================================================

class HilbertPolyaAnalyzer:
    """
    Finite-dimensional diagnostics for Hilbert--Pólya-style candidates.

    Notes
    -----
    These diagnostics test necessary numerical properties such as
    Hermiticity and spectral reality. They do not establish a
    Hilbert--Pólya correspondence or prove the Riemann Hypothesis.
    """

    def __init__(
        self,
        operator: LinearOperator,
    ):
        operator = _validate_operator(operator)

        object.__setattr__(self, "operator", operator)
        object.__setattr__(
            self,
            "spectrum",
            SpectralAnalyzer(operator),
        )

    def is_candidate_self_adjoint(
        self,
        tol: float = 1e-10,
    ) -> bool:
        """
        Check numerical Hermiticity.
        """

        tol = require_positive_real(
            tol,
            name="tol",
        )

        return self.operator.is_hermitian(
            tol=tol
        )

    def real_spectrum_defect(self) -> float:
        """
        Measure Euclidean imaginary leakage of the spectrum.
        """

        eigenvalues = self.spectrum.eigenvalues()

        return float(
            np.linalg.norm(
                eigenvalues.imag
            )
        )

    def summary(self) -> dict:
        """
        Return Hilbert--Pólya-style diagnostics.
        """

        eigenvalues = self.spectrum.eigenvalues()

        return {
            "operator": self.operator.name,
            "shape": self.operator.shape,
            "is_hermitian":
                self.operator.is_hermitian(),
            "real_spectrum_defect":
                self.real_spectrum_defect(),
            "num_eigenvalues": int(
                len(eigenvalues)
            ),
        }
