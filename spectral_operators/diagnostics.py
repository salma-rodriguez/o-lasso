"""
spectral_operators.diagnostics
==============================

Diagnostic tools for spectral operator models.

This module provides reusable summaries and comparisons across
algebraic, spectral, geometric, stability, and zeta-oriented
properties.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import OperatorError
from .core.utilities import require_same_shape
from .geometry import GeometryAnalyzer
from .spectrum import SpectralAnalyzer
from .zeta import HilbertPolyaAnalyzer, SpectralZeta


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
# Operator Diagnostics
# ===========================================================================

class OperatorDiagnostics:
    """
    General diagnostics for a single LinearOperator.
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

    def algebra_summary(self) -> dict:
        """
        Return algebraic and norm diagnostics.
        """

        operator = self.operator

        return {
            "name": operator.name,
            "shape": operator.shape,
            "field": operator.field.value,
            "dtype": str(operator.dtype),
            "rank": operator.rank,
            "trace": (
                operator.trace
                if operator.is_square
                else None
            ),
            "det": (
                operator.det
                if operator.is_square
                else None
            ),
            "cond": (
                operator.cond
                if operator.is_square
                else None
            ),
            "frobenius_norm":
                operator.norm("fro"),
            "spectral_norm":
                operator.norm("spectral"),
            "nuclear_norm":
                operator.norm("nuc"),
        }

    def geometry_summary(self) -> dict:
        """
        Return combined geometry diagnostics.
        """

        return GeometryAnalyzer(
            self.operator
        ).summary()

    def hilbert_polya_summary(self) -> dict:
        """
        Return Hilbert--Pólya-style diagnostics when applicable.
        """

        if not self.operator.is_square:
            return {}

        return HilbertPolyaAnalyzer(
            self.operator
        ).summary()

    def spectral_summary(self) -> dict:
        """
        Return eigenvalue and spacing diagnostics.
        """

        if not self.operator.is_square:
            return {
                "num_eigenvalues": None,
                "spacing_summary": None,
            }

        analyzer = SpectralAnalyzer(
            self.operator
        )
        statistics = analyzer.statistics()

        return {
            "num_eigenvalues": int(
                len(analyzer.eigenvalues())
            ),
            "spacing_summary":
                statistics.summary(),
        }

    def summary(self) -> dict:
        """
        Return the complete operator diagnostic summary.
        """

        return {
            "algebra": self.algebra_summary(),
            "spectral": self.spectral_summary(),
            "geometry": self.geometry_summary(),
            "hilbert_polya":
                self.hilbert_polya_summary(),
        }


# ===========================================================================
# Comparative Diagnostics
# ===========================================================================

class ComparativeDiagnostics:
    """
    Diagnostics for comparing two or more operators.
    """

    def __init__(
        self,
        operators,
    ):
        operator_tuple = tuple(operators)

        if not operator_tuple:
            raise OperatorError(
                "operators cannot be empty."
            )

        if not all(
            isinstance(operator, LinearOperator)
            for operator in operator_tuple
        ):
            raise OperatorError(
                "all inputs must be LinearOperator instances."
            )

        object.__setattr__(
            self,
            "operators",
            operator_tuple,
        )

    def norm_table(
        self,
        kind: str = "fro",
    ) -> dict:
        """
        Return operator norms keyed by operator name.
        """

        return {
            operator.name:
                operator.norm(kind)
            for operator in self.operators
        }

    def pairwise_distances(
        self,
        kind: str = "fro",
    ) -> dict:
        """
        Return pairwise operator distances.

        Shape-incompatible pairs are represented by ``None``.
        """

        distances = {}

        for index, left in enumerate(
            self.operators
        ):
            for right in self.operators[
                index + 1:
            ]:
                key = (
                    left.name,
                    right.name,
                )

                if left.shape != right.shape:
                    distances[key] = None
                    continue

                distances[key] = (
                    left - right
                ).norm(kind)

        return distances

    def rank_table(self) -> dict:
        """
        Return numerical ranks keyed by operator name.
        """

        return {
            operator.name:
                operator.rank
            for operator in self.operators
        }

    def summary(self) -> dict:
        """
        Return comparative diagnostics.
        """

        return {
            "num_operators": int(
                len(self.operators)
            ),
            "names": tuple(
                operator.name
                for operator in self.operators
            ),
            "frobenius_norms":
                self.norm_table("fro"),
            "spectral_norms":
                self.norm_table("spectral"),
            "ranks":
                self.rank_table(),
            "pairwise_frobenius_distances":
                self.pairwise_distances("fro"),
        }


# ===========================================================================
# Stability Diagnostics
# ===========================================================================

class StabilityDiagnostics:
    """
    Numerical and spectral stability diagnostics.
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

    def condition_number(
        self,
    ) -> float | None:
        """
        Return the condition number for square operators.
        """

        if not self.operator.is_square:
            return None

        return self.operator.cond

    def normality_defect(
        self,
    ) -> float | None:
        """
        Return the Frobenius norm of

            A†A - AA†.
        """

        if not self.operator.is_square:
            return None

        matrix = self.operator.matrix

        defect = (
            matrix.conj().T @ matrix
            - matrix @ matrix.conj().T
        )

        return float(
            np.linalg.norm(
                defect,
                ord="fro",
            )
        )

    def real_spectrum_defect(
        self,
    ) -> float | None:
        """
        Return imaginary spectral leakage.
        """

        if not self.operator.is_square:
            return None

        eigenvalues = (
            self.operator.eigenvalues()
        )

        return float(
            np.linalg.norm(
                eigenvalues.imag
            )
        )

    def spectral_radius(
        self,
    ) -> float | None:
        """
        Return the maximum eigenvalue magnitude.
        """

        if not self.operator.is_square:
            return None

        eigenvalues = (
            self.operator.eigenvalues()
        )

        if eigenvalues.size == 0:
            return 0.0

        return float(
            np.max(
                np.abs(eigenvalues)
            )
        )

    def summary(self) -> dict:
        """
        Return stability diagnostics.
        """

        return {
            "condition_number":
                self.condition_number(),
            "spectral_radius":
                self.spectral_radius(),
            "real_spectrum_defect":
                self.real_spectrum_defect(),
            "normality_defect":
                self.normality_defect(),
        }


# ===========================================================================
# Research Report
# ===========================================================================

class DiagnosticReport:
    """
    High-level report for operator experiments.
    """

    def __init__(
        self,
        operator: LinearOperator,
        *,
        include_zeta: bool = False,
    ):
        object.__setattr__(
            self,
            "operator",
            _validate_operator(operator),
        )
        object.__setattr__(
            self,
            "include_zeta",
            bool(include_zeta),
        )

    def generate(self) -> dict:
        """
        Generate the diagnostic report.
        """

        report = {
            "operator": self.operator.name,
            "diagnostics":
                OperatorDiagnostics(
                    self.operator
                ).summary(),
            "stability":
                StabilityDiagnostics(
                    self.operator
                ).summary(),
        }

        if (
            self.include_zeta
            and self.operator.is_square
        ):
            report["spectral_zeta"] = (
                SpectralZeta(
                    self.operator
                ).summary()
            )

        return report
