"""
stochastic_operators.ergodic
============================

Ergodic analysis for finite-state discrete-time Markov operators and
continuous-time Markov generators.

The module provides structural and numerical diagnostics for:

- irreducibility,
- communicating classes,
- periodicity and aperiodicity,
- ergodicity,
- spectral gaps,
- convergence to stationarity,
- total-variation error,
- Cesàro averages for discrete-time chains.

For a finite discrete-time Markov chain, ergodicity is taken to mean
irreducibility together with aperiodicity.

For a finite conservative continuous-time Markov chain, irreducibility
is sufficient for ergodicity.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from operator_core import (
    DimensionMismatchError,
    OperatorError,
    readonly_array,
)

from .generators import MarkovGenerator
from .markov import MarkovOperator
from .stationary import StationaryAnalyzer


# ===========================================================================
# Supported Models
# ===========================================================================

_SUPPORTED_MODEL_TYPES = (
    MarkovOperator,
    MarkovGenerator,
)


# ===========================================================================
# Validation Helpers
# ===========================================================================

def _validate_model(
    model,
) -> MarkovOperator | MarkovGenerator:
    """
    Validate and return a supported stochastic model.
    """

    if not isinstance(
        model,
        _SUPPORTED_MODEL_TYPES,
    ):
        raise OperatorError(
            "model must be a MarkovOperator or MarkovGenerator."
        )

    return model


def _validate_nonnegative_integer(
    value,
    *,
    name: str,
    allow_zero: bool = True,
) -> int:
    """
    Validate a nonnegative integer parameter.
    """

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (int, np.integer),
        )
    ):
        raise OperatorError(
            f"{name} must be an integer."
        )

    result = int(value)

    lower_bound = (
        0 if allow_zero else 1
    )

    if result < lower_bound:
        qualifier = (
            "nonnegative"
            if allow_zero
            else "positive"
        )

        raise OperatorError(
            f"{name} must be {qualifier}."
        )

    return result


def _validate_nonnegative_scalar(
    value,
    *,
    name: str,
) -> float:
    """
    Validate a finite nonnegative scalar.
    """

    if (
        isinstance(value, bool)
        or not np.isscalar(value)
    ):
        raise OperatorError(
            f"{name} must be a nonnegative scalar."
        )

    result = float(value)

    if (
        not np.isfinite(result)
        or result < 0.0
    ):
        raise OperatorError(
            f"{name} must be finite and nonnegative."
        )

    return result


def _validate_distribution(
    distribution,
    *,
    dimension: int,
    tol: float,
) -> np.ndarray:
    """
    Validate and return a probability distribution.
    """

    values = np.asarray(
        distribution,
        dtype=float,
    )

    if values.ndim != 1:
        raise OperatorError(
            "distribution must be one-dimensional."
        )

    if len(values) != dimension:
        raise DimensionMismatchError(
            "distribution dimension does not match the stochastic model."
        )

    if not np.all(
        np.isfinite(values)
    ):
        raise OperatorError(
            "distribution values must be finite."
        )

    if np.any(
        values < -tol
    ):
        raise OperatorError(
            "distribution values must be nonnegative."
        )

    if not np.isclose(
        np.sum(values),
        1.0,
        atol=tol,
        rtol=tol,
    ):
        raise OperatorError(
            "distribution must sum to one."
        )

    result = np.array(
        values,
        dtype=float,
        copy=True,
    )

    result[
        np.abs(result) <= tol
    ] = 0.0

    return result


# ===========================================================================
# Ergodic Analyzer
# ===========================================================================

class ErgodicAnalyzer:
    """
    Analyze ergodic properties of a finite-state stochastic model.

    Parameters
    ----------
    model
        A MarkovOperator or MarkovGenerator.

    tol
        Optional numerical tolerance. When omitted, the tolerance of the
        stochastic model is used.
    """

    def __init__(
        self,
        model,
        *,
        tol: float | None = None,
    ):
        model = _validate_model(
            model
        )

        if tol is None:
            tolerance = float(
                model.tol
            )
        else:
            tolerance = (
                _validate_nonnegative_scalar(
                    tol,
                    name="tol",
                )
            )

        object.__setattr__(
            self,
            "model",
            model,
        )

        object.__setattr__(
            self,
            "tol",
            tolerance,
        )

    # -----------------------------------------------------------------------
    # Basic Properties
    # -----------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """
        Return the number of states.
        """

        return self.model.dimension

    @property
    def states(self) -> tuple:
        """
        Return the state labels.
        """

        return self.model.states

    @property
    def is_discrete_time(self) -> bool:
        """
        Return whether the model is discrete-time.
        """

        return isinstance(
            self.model,
            MarkovOperator,
        )

    @property
    def is_continuous_time(self) -> bool:
        """
        Return whether the model is continuous-time.
        """

        return isinstance(
            self.model,
            MarkovGenerator,
        )

    @property
    def stationary_analyzer(
        self,
    ) -> StationaryAnalyzer:
        """
        Return a stationary analyzer for the same model.
        """

        return StationaryAnalyzer(
            self.model,
            tol=self.tol,
        )

    # -----------------------------------------------------------------------
    # Convention-Independent Matrix Representation
    # -----------------------------------------------------------------------

    def row_oriented_matrix(
        self,
    ) -> np.ndarray:
        """
        Return the transition or generator matrix in row convention.
        """

        matrix = self.model.matrix

        if self.is_discrete_time:
            row_oriented = (
                matrix
                if self.model.is_row_stochastic
                else matrix.T
            )
        else:
            row_oriented = (
                matrix
                if self.model.is_row_generator
                else matrix.T
            )

        return readonly_array(
            row_oriented,
            name="row-oriented matrix",
            ndim=2,
        )

    def adjacency_matrix(
        self,
    ) -> np.ndarray:
        """
        Return the directed adjacency matrix of allowed state transitions.

        Self-loops are retained when they have positive transition
        probability or positive off-diagonal-equivalent activity.
        """

        matrix = self.row_oriented_matrix()

        if self.is_discrete_time:
            adjacency = (
                matrix > self.tol
            )
        else:
            adjacency = (
                matrix > self.tol
            )

            np.fill_diagonal(
                adjacency,
                False,
            )

        adjacency = np.asarray(
            adjacency,
            dtype=bool,
        )

        return readonly_array(
            adjacency,
            name="ergodic adjacency matrix",
            ndim=2,
        )

    # -----------------------------------------------------------------------
    # Communicating Classes and Irreducibility
    # -----------------------------------------------------------------------

    def communicating_classes(
        self,
    ) -> tuple[tuple, ...]:
        """
        Return strongly connected communicating classes.

        State labels, rather than integer indices, are returned.
        """

        graph = csr_matrix(
            self.adjacency_matrix().astype(
                np.int8
            )
        )

        number_of_classes, labels = (
            connected_components(
                graph,
                directed=True,
                connection="strong",
                return_labels=True,
            )
        )

        classes = []

        for class_index in range(
            number_of_classes
        ):
            indices = np.flatnonzero(
                labels == class_index
            )

            state_class = tuple(
                self.states[index]
                for index in indices
            )

            classes.append(
                state_class
            )

        classes.sort(
            key=lambda state_class: min(
                self.model.state_index(state)
                for state in state_class
            )
        )

        return tuple(classes)

    def is_irreducible(
        self,
    ) -> bool:
        """
        Return whether every state communicates with every other state.
        """

        return (
            len(
                self.communicating_classes()
            )
            == 1
        )

    # -----------------------------------------------------------------------
    # Discrete-Time Periodicity
    # -----------------------------------------------------------------------

    def period(
        self,
    ) -> int | None:
        """
        Return the period of an irreducible discrete-time chain.

        Returns
        -------
        int or None
            The common state period for an irreducible discrete-time chain.
            ``None`` is returned for continuous-time models or reducible
            discrete-time chains.

        Notes
        -----
        The period is computed from graph-depth differences using the
        greatest common divisor characterization of cycle lengths.
        """

        if not self.is_discrete_time:
            return None

        if not self.is_irreducible():
            return None

        adjacency = self.adjacency_matrix()
        dimension = self.dimension

        distances = np.full(
            dimension,
            -1,
            dtype=int,
        )

        root = 0
        distances[root] = 0

        queue = [root]
        position = 0

        while position < len(queue):
            source = queue[position]
            position += 1

            neighbors = np.flatnonzero(
                adjacency[source]
            )

            for target in neighbors:
                if distances[target] == -1:
                    distances[target] = (
                        distances[source]
                        + 1
                    )
                    queue.append(
                        int(target)
                    )

        cycle_gcd = 0

        for source in range(
            dimension
        ):
            neighbors = np.flatnonzero(
                adjacency[source]
            )

            for target in neighbors:
                difference = (
                    distances[source]
                    + 1
                    - distances[target]
                )

                cycle_gcd = gcd(
                    cycle_gcd,
                    abs(
                        int(difference)
                    ),
                )

        return max(
            cycle_gcd,
            1,
        )

    def is_aperiodic(
        self,
    ) -> bool:
        """
        Return whether the stochastic model is aperiodic.

        Every finite continuous-time Markov chain is treated as aperiodic in
        the semigroup sense. For discrete-time models, the chain must be
        irreducible and have period one.
        """

        if self.is_continuous_time:
            return True

        chain_period = self.period()

        return (
            chain_period == 1
        )

    # -----------------------------------------------------------------------
    # Ergodicity
    # -----------------------------------------------------------------------

    def is_ergodic(
        self,
    ) -> bool:
        """
        Return whether the stochastic model is ergodic.

        For finite discrete-time chains:

            ergodic = irreducible and aperiodic.

        For finite conservative continuous-time chains:

            ergodic = irreducible.
        """

        if self.is_discrete_time:
            return (
                self.is_irreducible()
                and self.is_aperiodic()
            )

        return self.is_irreducible()

    # -----------------------------------------------------------------------
    # Spectral Diagnostics
    # -----------------------------------------------------------------------

    def eigenvalues(
        self,
    ) -> np.ndarray:
        """
        Return the eigenvalues of the transition matrix or generator.
        """

        values = np.linalg.eigvals(
            self.model.matrix
        )

        return readonly_array(
            values,
            name="ergodic eigenvalues",
            ndim=1,
        )

    def spectral_gap(
        self,
    ) -> float:
        """
        Return the discrete-time or continuous-time spectral gap.

        Discrete time
        -------------
        The gap is

            1 - max{|lambda| : lambda != 1}.

        Continuous time
        ---------------
        The gap is

            -max{Re(lambda) : lambda != 0}.

        A zero gap is returned when no nonstationary eigenvalue exists.
        Small negative values caused by floating-point noise are clipped
        to zero.
        """

        eigenvalues = np.asarray(
            self.eigenvalues()
        )

        if self.is_discrete_time:
            nonstationary = eigenvalues[
                ~np.isclose(
                    eigenvalues,
                    1.0,
                    atol=self.tol,
                    rtol=self.tol,
                )
            ]

            if nonstationary.size == 0:
                return 0.0

            gap = (
                1.0
                - float(
                    np.max(
                        np.abs(
                            nonstationary
                        )
                    )
                )
            )

        else:
            nonstationary = eigenvalues[
                ~np.isclose(
                    eigenvalues,
                    0.0,
                    atol=self.tol,
                    rtol=self.tol,
                )
            ]

            if nonstationary.size == 0:
                return 0.0

            gap = -float(
                np.max(
                    np.real(
                        nonstationary
                    )
                )
            )

        return max(
            gap,
            0.0,
        )

    def has_positive_spectral_gap(
        self,
    ) -> bool:
        """
        Return whether the numerical spectral gap is positive.
        """

        return (
            self.spectral_gap()
            > self.tol
        )

    # -----------------------------------------------------------------------
    # Distribution Evolution
    # -----------------------------------------------------------------------

    def evolved_distribution(
        self,
        initial_distribution,
        *,
        steps: int = 1,
        time: float = 1.0,
    ) -> np.ndarray:
        """
        Evolve a distribution using the underlying stochastic model.
        """

        distribution = (
            _validate_distribution(
                initial_distribution,
                dimension=self.dimension,
                tol=self.tol,
            )
        )

        if self.is_discrete_time:
            validated_steps = (
                _validate_nonnegative_integer(
                    steps,
                    name="steps",
                )
            )

            return self.model.evolve_distribution(
                distribution,
                steps=validated_steps,
            )

        validated_time = (
            _validate_nonnegative_scalar(
                time,
                name="time",
            )
        )

        return self.model.evolve_distribution(
            distribution,
            t=validated_time,
        )

    def stationary_distribution(
        self,
    ) -> np.ndarray:
        """
        Return one stationary distribution.
        """

        return (
            self.stationary_analyzer
            .stationary_distribution()
        )

    # -----------------------------------------------------------------------
    # Convergence Diagnostics
    # -----------------------------------------------------------------------

    def total_variation_distance(
        self,
        distribution,
        reference=None,
    ) -> float:
        """
        Return the total-variation distance between two distributions.

        If ``reference`` is omitted, the stationary distribution is used.
        """

        values = _validate_distribution(
            distribution,
            dimension=self.dimension,
            tol=self.tol,
        )

        if reference is None:
            target = (
                self.stationary_distribution()
            )
        else:
            target = _validate_distribution(
                reference,
                dimension=self.dimension,
                tol=self.tol,
            )

        return float(
            0.5
            * np.linalg.norm(
                values - target,
                ord=1,
            )
        )

    def convergence_error(
        self,
        initial_distribution,
        *,
        steps: int = 1000,
        time: float = 100.0,
    ) -> float:
        """
        Return the total-variation error after discrete steps or elapsed time.
        """

        evolved = self.evolved_distribution(
            initial_distribution,
            steps=steps,
            time=time,
        )

        return self.total_variation_distance(
            evolved
        )

    def has_converged(
        self,
        initial_distribution,
        *,
        steps: int = 1000,
        time: float = 100.0,
        threshold: float | None = None,
    ) -> bool:
        """
        Check whether evolution is numerically close to stationarity.
        """

        if threshold is None:
            convergence_threshold = max(
                self.tol,
                np.sqrt(
                    np.finfo(float).eps
                ),
            )
        else:
            convergence_threshold = (
                _validate_nonnegative_scalar(
                    threshold,
                    name="threshold",
                )
            )

        return (
            self.convergence_error(
                initial_distribution,
                steps=steps,
                time=time,
            )
            <= convergence_threshold
        )

    # -----------------------------------------------------------------------
    # Cesàro Averages
    # -----------------------------------------------------------------------

    def cesaro_average(
        self,
        initial_distribution,
        *,
        steps: int,
    ) -> np.ndarray:
        """
        Return the discrete-time Cesàro average of evolved distributions.

        The returned quantity is

            (1 / (steps + 1))
            sum_{k=0}^{steps} mu P^k.

        Cesàro averaging can converge even when an irreducible chain is
        periodic and its ordinary iterates do not converge.
        """

        if not self.is_discrete_time:
            raise OperatorError(
                "Cesàro averaging is implemented only for "
                "discrete-time Markov operators."
            )

        validated_steps = (
            _validate_nonnegative_integer(
                steps,
                name="steps",
            )
        )

        distribution = (
            _validate_distribution(
                initial_distribution,
                dimension=self.dimension,
                tol=self.tol,
            )
        )

        current = np.array(
            distribution,
            dtype=float,
            copy=True,
        )

        total = np.array(
            current,
            dtype=float,
            copy=True,
        )

        transition = (
            self.row_oriented_matrix()
        )

        for _ in range(
            validated_steps
        ):
            current = (
                current @ transition
            )

            total += current

        average = (
            total
            / float(
                validated_steps + 1
            )
        )

        average[
            np.abs(average) <= self.tol
        ] = 0.0

        average /= np.sum(
            average
        )

        return readonly_array(
            average,
            name="Cesaro average",
            ndim=1,
        )

    def cesaro_error(
        self,
        initial_distribution,
        *,
        steps: int,
    ) -> float:
        """
        Return the Cesàro-average total-variation error.
        """

        average = self.cesaro_average(
            initial_distribution,
            steps=steps,
        )

        return self.total_variation_distance(
            average
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return structural and spectral ergodic diagnostics.
        """

        chain_period = self.period()

        return {
            "model": self.model.name,
            "model_type": (
                "discrete_time"
                if self.is_discrete_time
                else "continuous_time"
            ),
            "dimension": self.dimension,
            "states": self.states,
            "communicating_classes":
                self.communicating_classes(),
            "irreducible":
                self.is_irreducible(),
            "period":
                chain_period,
            "aperiodic":
                self.is_aperiodic(),
            "ergodic":
                self.is_ergodic(),
            "stationary_dimension":
                self.stationary_analyzer
                .stationary_dimension(),
            "unique_stationary_distribution":
                self.stationary_analyzer
                .is_unique(),
            "spectral_gap":
                self.spectral_gap(),
            "positive_spectral_gap":
                self.has_positive_spectral_gap(),
        }
