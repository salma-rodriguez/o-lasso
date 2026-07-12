"""
spectral_operators.adelic
=========================

Adelic-style constructions for spectral operator models.

This module provides labeled local components, adelic systems,
global operator assembly, and diagnostics for weighted local-to-global
operator constructions.
"""

from __future__ import annotations

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import (
    DimensionMismatchError,
    OperatorError,
)
from .core.utilities import require_same_shape
from .operators import AdelicOperator
from .weights import AdelicWeight


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _validate_system(system: "AdelicSystem") -> "AdelicSystem":
    """
    Validate and return an AdelicSystem.
    """

    if not isinstance(system, AdelicSystem):
        raise OperatorError(
            "system must be an AdelicSystem."
        )

    return system


# ===========================================================================
# Local Components
# ===========================================================================

class LocalComponent:
    """
    Labeled local operator component.

    Parameters
    ----------
    label
        Identifier for the local component, such as a prime,
        place, scale, or region.

    operator
        LinearOperator associated with the label.
    """

    def __init__(
        self,
        label,
        operator: LinearOperator,
    ):
        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        object.__setattr__(self, "label", label)
        object.__setattr__(self, "operator", operator)

    @property
    def matrix(self) -> np.ndarray:
        """
        Return the local operator matrix.
        """

        return self.operator.matrix

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the local operator shape.
        """

        return self.operator.shape

    def as_tuple(
        self,
    ) -> tuple[object, LinearOperator]:
        """
        Return ``(label, operator)``.
        """

        return self.label, self.operator


# ===========================================================================
# Adelic System
# ===========================================================================

class AdelicSystem:
    """
    Collection of compatible local components and associated weights.
    """

    def __init__(
        self,
        components,
        *,
        weights=None,
        normalize: bool = True,
    ):
        component_tuple = tuple(components)

        if not component_tuple:
            raise OperatorError(
                "AdelicSystem requires at least one component."
            )

        if not all(
            isinstance(component, LocalComponent)
            for component in component_tuple
        ):
            raise OperatorError(
                "components must be LocalComponent instances."
            )

        reference = component_tuple[0]

        for component in component_tuple[1:]:
            require_same_shape(
                reference.operator,
                component.operator,
                left_name=str(reference.label),
                right_name=str(component.label),
            )

        labels = tuple(
            component.label
            for component in component_tuple
        )

        if len(set(labels)) != len(labels):
            raise OperatorError(
                "component labels must be unique."
            )

        if weights is None:
            weight_system = AdelicWeight(
                labels=labels,
                normalize=normalize,
            )

        elif isinstance(weights, AdelicWeight):
            weight_system = weights

        else:
            weight_system = AdelicWeight(
                labels=labels,
                weights=weights,
                normalize=normalize,
            )

        if tuple(weight_system.labels) != labels:
            raise DimensionMismatchError(
                "weight labels must match component labels."
            )

        object.__setattr__(
            self,
            "components",
            component_tuple,
        )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "weights",
            weight_system,
        )
        object.__setattr__(
            self,
            "shape",
            reference.shape,
        )

    @classmethod
    def from_operators(
        cls,
        operators,
        *,
        labels=None,
        weights=None,
        normalize: bool = True,
    ) -> "AdelicSystem":
        """
        Construct a system directly from LinearOperator objects.
        """

        operator_tuple = tuple(operators)

        if not operator_tuple:
            raise OperatorError(
                "operators cannot be empty."
            )

        if labels is None:
            label_tuple = tuple(
                range(len(operator_tuple))
            )
        else:
            label_tuple = tuple(labels)

        if len(label_tuple) != len(operator_tuple):
            raise DimensionMismatchError(
                "labels must match the number of operators."
            )

        components = tuple(
            LocalComponent(label, operator)
            for label, operator in zip(
                label_tuple,
                operator_tuple,
            )
        )

        return cls(
            components,
            weights=weights,
            normalize=normalize,
        )

    def local_operators(
        self,
    ) -> tuple[LinearOperator, ...]:
        """
        Return the local operators as an immutable tuple.
        """

        return tuple(
            component.operator
            for component in self.components
        )

    def weight_array(self) -> np.ndarray:
        """
        Return a mutable copy of the system weights.
        """

        return self.weights.as_array()


# ===========================================================================
# Adelic Builder
# ===========================================================================

class AdelicBuilder:
    """
    Builder for assembling adelic-style global operators.
    """

    def __init__(
        self,
        system: AdelicSystem,
    ):
        object.__setattr__(
            self,
            "system",
            _validate_system(system),
        )

    def build(
        self,
        *,
        name: str | None = None,
    ) -> AdelicOperator:
        """
        Build the global weighted AdelicOperator.
        """

        return AdelicOperator(
            self.system.local_operators(),
            weights=self.system.weight_array(),
            labels=self.system.labels,
            normalize=False,
            name=name or "AdelicOperator",
        )


# ===========================================================================
# Adelic Analyzer
# ===========================================================================

class AdelicAnalyzer:
    """
    Diagnostics for adelic-style operator systems.
    """

    def __init__(
        self,
        system: AdelicSystem,
    ):
        object.__setattr__(
            self,
            "system",
            _validate_system(system),
        )

    def component_norms(
        self,
        kind: str = "fro",
    ) -> dict:
        """
        Return norms of the unweighted local components.
        """

        return {
            component.label:
                component.operator.norm(kind)
            for component in self.system.components
        }

    def summary(
        self,
        kind: str = "fro",
    ) -> dict:
        """
        Return a complete adelic diagnostic summary.
        """

        return {
            "shape": self.system.shape,
            "labels": self.system.labels,
            "component_norms":
                self.component_norms(kind),
            "weighted_component_norms":
                self.weighted_component_norms(kind),
            "weights":
                self.weight_summary(),
        }

    def weight_summary(self) -> dict:
        """
        Return information about the adelic weight system.
        """

        weights = self.system.weight_array()

        return {
            "labels": self.system.labels,
            "weights": tuple(
                weights.tolist()
            ),
            "sum_abs_weights": float(
                np.sum(np.abs(weights))
            ),
            "num_components": int(
                len(weights)
            ),
        }

    def weighted_component_norms(
        self,
        kind: str = "fro",
    ) -> dict:
        """
        Return absolute-weight-scaled local component norms.
        """

        weights = self.system.weight_array()

        return {
            component.label: float(
                abs(weight)
                * component.operator.norm(kind)
            )
            for component, weight in zip(
                self.system.components,
                weights,
            )
        }
