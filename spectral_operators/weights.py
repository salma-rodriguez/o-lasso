"""
spectral_operators.weights
==========================

Weight constructions for operator-theoretic models.

This module defines diagonal weight operators, position-dependent
weight profiles, polynomial and exponential weights, and scalar
weight systems for prime-indexed and adelic constructions.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import DimensionMismatchError, OperatorError
from .core.utilities import (
    as_one_dimensional_array,
    normalize_l1,
    normalize_max_abs,
    readonly_array,
    require_positive_integer,
    require_positive_real,
)


# ===========================================================================
# Base Weight
# ===========================================================================

class WeightOperator(LinearOperator):
    """
    Base diagonal weight operator.

    Parameters
    ----------
    weights
        One-dimensional weight values.

    dtype
        Optional NumPy dtype.

    name
        Human-readable operator name.
    """

    def __init__(
        self,
        weights,
        *,
        dtype=None,
        name: str = "WeightOperator",
    ):
        weight_array = as_one_dimensional_array(
            weights,
            name="weights",
            dtype=dtype,
            copy=True,
        )

        super().__init__(
            matrix=np.diag(weight_array),
            name=name,
            metadata={
                "operator": "weight",
                "dimension": len(weight_array),
            },
        )

        frozen_weights = readonly_array(
            weight_array,
            name="weights",
            ndim=1,
        )

        object.__setattr__(self, "weights", frozen_weights)


# ===========================================================================
# Position Weights
# ===========================================================================

class PositionWeight(WeightOperator):
    """
    Position-dependent diagonal weight operator.

    Constructs

        w(x) = offset + scale |x|^power

    on a uniform grid over ``[-L, L]``.
    """

    def __init__(
        self,
        N: int,
        L: float,
        *,
        power: float = 2.0,
        scale: float = 1.0,
        offset: float = 1.0,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):
        N = require_positive_integer(N, name="N")
        L = require_positive_real(L, name="L")

        grid = np.linspace(-L, L, N, dtype=dtype)
        weights = offset + scale * np.abs(grid) ** power

        if normalize:
            weights = normalize_max_abs(
                weights,
                name="position weights",
            )

        super().__init__(
            weights,
            dtype=dtype,
            name=name or "PositionWeight",
        )

        object.__setattr__(
            self,
            "grid",
            readonly_array(grid, name="grid", ndim=1),
        )
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Polynomial Weights
# ===========================================================================

class PolynomialWeight(WeightOperator):
    """
    Polynomial diagonal weight operator.

    Constructs

        w(x) = c_0 + c_1 x + c_2 x^2 + ...

    on a uniform grid over ``[-L, L]``.
    """

    def __init__(
        self,
        N: int,
        L: float,
        coefficients,
        *,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):
        N = require_positive_integer(N, name="N")
        L = require_positive_real(L, name="L")

        coefficients_array = as_one_dimensional_array(
            coefficients,
            name="coefficients",
            dtype=dtype,
            copy=True,
        )

        grid = np.linspace(-L, L, N, dtype=dtype)

        # np.polynomial.polynomial.polyval uses coefficients in ascending
        # order: c0, c1, c2, ...
        weights = np.polynomial.polynomial.polyval(
            grid,
            coefficients_array,
        )

        if normalize:
            weights = normalize_max_abs(
                weights,
                name="polynomial weights",
            )

        super().__init__(
            weights,
            dtype=dtype,
            name=name or "PolynomialWeight",
        )

        object.__setattr__(
            self,
            "grid",
            readonly_array(grid, name="grid", ndim=1),
        )
        object.__setattr__(
            self,
            "coefficients",
            readonly_array(
                coefficients_array,
                name="coefficients",
                ndim=1,
            ),
        )
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Exponential Weights
# ===========================================================================

class ExponentialWeight(WeightOperator):
    """
    Exponential diagonal weight operator.

    Constructs

        w(x) = offset + scale exp(rate |x|^power)

    on a uniform grid over ``[-L, L]``.
    """

    def __init__(
        self,
        N: int,
        L: float,
        *,
        rate: float = -1.0,
        power: float = 2.0,
        scale: float = 1.0,
        offset: float = 0.0,
        normalize: bool = False,
        dtype=float,
        name: str | None = None,
    ):
        N = require_positive_integer(N, name="N")
        L = require_positive_real(L, name="L")

        grid = np.linspace(-L, L, N, dtype=dtype)
        weights = offset + scale * np.exp(
            rate * np.abs(grid) ** power
        )

        if not np.all(np.isfinite(weights)):
            raise OperatorError(
                "Exponential weight construction produced non-finite values."
            )

        if normalize:
            weights = normalize_max_abs(
                weights,
                name="exponential weights",
            )

        super().__init__(
            weights,
            dtype=dtype,
            name=name or "ExponentialWeight",
        )

        object.__setattr__(
            self,
            "grid",
            readonly_array(grid, name="grid", ndim=1),
        )
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "normalize", normalize)


# ===========================================================================
# Prime / Adelic Weights
# ===========================================================================

class PrimeWeight:
    """
    Scalar weights associated with prime-indexed local operators.

    Notes
    -----
    This class validates that supplied values are greater than one,
    but it does not currently perform primality testing.
    """

    _SUPPORTED_RULES = {
        "inverse",
        "log_inverse",
        "log",
        "uniform",
    }

    def __init__(
        self,
        primes,
        *,
        rule: str = "inverse",
        normalize: bool = True,
    ):
        prime_array = as_one_dimensional_array(
            primes,
            name="primes",
            dtype=float,
            copy=True,
        )

        if np.any(~np.isfinite(prime_array)):
            raise OperatorError(
                "prime values must be finite."
            )

        if np.any(prime_array <= 1):
            raise OperatorError(
                "prime values must be greater than 1."
            )

        if rule not in self._SUPPORTED_RULES:
            raise OperatorError(
                "rule must be one of: "
                "'inverse', 'log_inverse', 'log', or 'uniform'."
            )

        if rule == "inverse":
            weights = 1.0 / prime_array

        elif rule == "log_inverse":
            weights = 1.0 / np.log(prime_array)

        elif rule == "log":
            weights = np.log(prime_array)

        else:
            weights = np.ones_like(prime_array)

        if normalize:
            weights = normalize_l1(
                weights,
                name="prime weights",
            )

        object.__setattr__(
            self,
            "primes",
            readonly_array(
                prime_array,
                name="primes",
                ndim=1,
            ),
        )
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "normalize", normalize)
        object.__setattr__(
            self,
            "weights",
            readonly_array(
                weights,
                name="weights",
                ndim=1,
            ),
        )

    def as_array(self) -> np.ndarray:
        """Return a mutable copy of the weight array."""
        return self.weights.copy()

    def as_dict(self) -> dict[int, float]:
        """Return a prime-to-weight mapping."""
        return {
            int(prime): float(weight)
            for prime, weight in zip(
                self.primes,
                self.weights,
            )
        }


class AdelicWeight:
    """
    General weight system for assembling labeled local components.
    """

    def __init__(
        self,
        labels,
        weights=None,
        *,
        normalize: bool = True,
    ):
        label_tuple = tuple(labels)

        if not label_tuple:
            raise OperatorError(
                "labels cannot be empty."
            )

        if len(set(label_tuple)) != len(label_tuple):
            raise OperatorError(
                "labels must be unique."
            )

        if weights is None:
            weight_array = np.ones(
                len(label_tuple),
                dtype=float,
            )
        else:
            weight_array = as_one_dimensional_array(
                weights,
                name="weights",
                dtype=float,
                copy=True,
            )

        if len(weight_array) != len(label_tuple):
            raise DimensionMismatchError(
                "weights must match the number of labels."
            )

        if normalize:
            weight_array = normalize_l1(
                weight_array,
                name="adelic weights",
            )

        object.__setattr__(self, "labels", label_tuple)
        object.__setattr__(self, "normalize", normalize)
        object.__setattr__(
            self,
            "weights",
            readonly_array(
                weight_array,
                name="weights",
                ndim=1,
            ),
        )

    @classmethod
    def from_primes(
        cls,
        primes,
        *,
        rule: str = "inverse",
        normalize: bool = True,
    ) -> "AdelicWeight":
        prime_weights = PrimeWeight(
            primes,
            rule=rule,
            normalize=normalize,
        )

        return cls(
            labels=tuple(
                int(prime)
                for prime in prime_weights.primes
            ),
            weights=prime_weights.weights,
            normalize=False,
        )

    def as_array(self) -> np.ndarray:
        """Return a mutable copy of the weight array."""
        return self.weights.copy()

    def as_dict(self) -> dict:
        """Return a label-to-weight mapping."""
        return {
            label: float(weight)
            for label, weight in zip(
                self.labels,
                self.weights,
            )
        }
