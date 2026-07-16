"""
stochastic_operators.markov
===========================

Finite-state discrete-time Markov operators.

This module extends StochasticOperator with labeled states,
transition queries, distribution evolution, trajectories,
reachability, communication, and absorbing-state diagnostics.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from operator_core import (
    DimensionMismatchError,
    OperatorError,
    readonly_array,
    require_nonnegative_integer,
)
from .operators import (
    StochasticConvention,
    StochasticOperator,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _validate_state_labels(
    states,
    *,
    dimension: int,
) -> tuple:
    """
    Validate and return an immutable tuple of state labels.
    """

    if states is None:
        return tuple(range(dimension))

    labels = tuple(states)

    if len(labels) != dimension:
        raise DimensionMismatchError(
            "number of state labels must match operator dimension."
        )

    if len(set(labels)) != len(labels):
        raise OperatorError(
            "state labels must be unique."
        )

    return labels


# ===========================================================================
# Markov Operator
# ===========================================================================

class MarkovOperator(StochasticOperator):
    """
    Finite-state discrete-time Markov transition operator.

    Parameters
    ----------
    matrix
        Square stochastic transition matrix.

    states
        Optional unique state labels. If omitted, integer labels
        ``0, 1, ..., n - 1`` are used.

    convention
        Row- or column-stochastic convention.

    tol
        Numerical tolerance used for stochastic validation.

    name
        Human-readable operator name.

    metadata
        Optional operator metadata.
    """

    def __init__(
        self,
        matrix,
        *,
        states=None,
        convention: StochasticConvention | str = StochasticConvention.ROW,
        tol: float = 1e-10,
        name: str = "MarkovOperator",
        metadata: dict | None = None,
    ):
        super().__init__(
            matrix,
            convention=convention,
            tol=tol,
            name=name,
            metadata=metadata,
        )

        state_labels = _validate_state_labels(
            states,
            dimension=self.dimension,
        )

        state_to_index = {
            state: index
            for index, state in enumerate(state_labels)
        }

        object.__setattr__(
            self,
            "states",
            state_labels,
        )
        object.__setattr__(
            self,
            "_state_to_index",
            state_to_index,
        )

        self.metadata.update({
            "operator": "markov",
            "states": state_labels,
        })

    # -----------------------------------------------------------------------
    # State Resolution
    # -----------------------------------------------------------------------

    def state_index(
        self,
        state,
    ) -> int:
        """
        Return the integer index associated with a state label.
        """

        try:
            return self._state_to_index[state]
        except KeyError as exc:
            raise OperatorError(
                f"unknown state label: {state!r}."
            ) from exc

    def state_label(
        self,
        index: int,
    ):
        """
        Return the state label associated with an integer index.
        """

        index = require_nonnegative_integer(
            index,
            name="index",
        )

        if index >= self.dimension:
            raise OperatorError(
                "state index is outside the valid range."
            )

        return self.states[index]

    # -----------------------------------------------------------------------
    # Transition Probabilities
    # -----------------------------------------------------------------------

    def transition_probability(
        self,
        source,
        target,
        *,
        steps: int = 1,
    ) -> float:
        """
        Return the probability of transitioning from ``source`` to
        ``target`` in exactly ``steps`` transitions.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        source_index = self.state_index(source)
        target_index = self.state_index(target)

        transition_matrix = np.linalg.matrix_power(
            self.matrix,
            steps,
        )

        if self.is_row_stochastic:
            probability = transition_matrix[
                source_index,
                target_index,
            ]
        else:
            probability = transition_matrix[
                target_index,
                source_index,
            ]

        return float(probability)

    def transition_matrix(
        self,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Return the transition matrix after ``steps`` transitions.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        matrix = np.linalg.matrix_power(
            self.matrix,
            steps,
        )

        return readonly_array(
            matrix,
            name="transition matrix",
            ndim=2,
        )

    # -----------------------------------------------------------------------
    # Distribution Evolution
    # -----------------------------------------------------------------------

    def evolve_distribution(
        self,
        distribution,
        *,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Evolve a probability distribution by a number of transitions.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        vector = np.asarray(
            distribution,
            dtype=float,
        )

        if vector.ndim != 1:
            raise OperatorError(
                "distribution must be one-dimensional."
            )

        if len(vector) != self.dimension:
            raise DimensionMismatchError(
                "distribution dimension does not match the Markov operator."
            )

        if not np.all(np.isfinite(vector)):
            raise OperatorError(
                "distribution values must be finite."
            )

        if np.any(vector < -self.tol):
            raise OperatorError(
                "distribution values must be nonnegative."
            )

        if not np.isclose(
            np.sum(vector),
            1.0,
            atol=self.tol,
            rtol=self.tol,
        ):
            raise OperatorError(
                "distribution must sum to one."
            )

        matrix = np.linalg.matrix_power(
            self.matrix,
            steps,
        )

        if self.is_row_stochastic:
            evolved = vector @ matrix
        else:
            evolved = matrix @ vector

        return readonly_array(
            evolved,
            name="evolved distribution",
            ndim=1,
        )

    def distribution_history(
        self,
        distribution,
        *,
        steps: int,
    ) -> np.ndarray:
        """
        Return the distribution at times ``0, 1, ..., steps``.

        The result has shape ``(steps + 1, dimension)``.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        initial = self.evolve_distribution(
            distribution,
            steps=0,
        )

        history = np.empty(
            (steps + 1, self.dimension),
            dtype=float,
        )
        history[0] = initial

        current = initial.copy()

        for step in range(1, steps + 1):
            current = self.apply_distribution(current)
            history[step] = current

        return readonly_array(
            history,
            name="distribution history",
            ndim=2,
        )

    # -----------------------------------------------------------------------
    # State Classification
    # -----------------------------------------------------------------------

    def is_absorbing(
        self,
        state,
    ) -> bool:
        """
        Return whether a state is absorbing.
        """

        index = self.state_index(state)

        if self.is_row_stochastic:
            probabilities = self.matrix[index, :]
        else:
            probabilities = self.matrix[:, index]

        expected = np.zeros(
            self.dimension,
            dtype=float,
        )
        expected[index] = 1.0

        return bool(
            np.allclose(
                probabilities,
                expected,
                atol=self.tol,
                rtol=self.tol,
            )
        )

    def absorbing_states(self) -> tuple:
        """
        Return all absorbing state labels.
        """

        return tuple(
            state
            for state in self.states
            if self.is_absorbing(state)
        )

    # -----------------------------------------------------------------------
    # Reachability and Communication
    # -----------------------------------------------------------------------

    def adjacency_matrix(self) -> np.ndarray:
        """
        Return the directed adjacency matrix induced by positive
        transition probabilities.
        """

        if self.is_row_stochastic:
            adjacency = self.matrix > self.tol
        else:
            adjacency = self.matrix.T > self.tol

        return readonly_array(
            adjacency,
            name="Markov adjacency matrix",
            ndim=2,
        )

    def reachable_states(
        self,
        source,
        *,
        include_source: bool = True,
    ) -> tuple:
        """
        Return states reachable from ``source`` through positive-probability
        transition paths.
        """

        source_index = self.state_index(source)
        adjacency = self.adjacency_matrix()

        visited = {source_index}
        queue = deque([source_index])

        while queue:
            current = queue.popleft()

            neighbors = np.flatnonzero(
                adjacency[current]
            )

            for neighbor in neighbors:
                neighbor = int(neighbor)

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if not include_source:
            visited.discard(source_index)

        return tuple(
            self.states[index]
            for index in range(self.dimension)
            if index in visited
        )

    def communicates(
        self,
        left,
        right,
    ) -> bool:
        """
        Return whether two states communicate.

        Two states communicate when each is reachable from the other.
        """

        left_reachable = set(
            self.reachable_states(left)
        )
        right_reachable = set(
            self.reachable_states(right)
        )

        return (
            right in left_reachable
            and left in right_reachable
        )

    def communicating_classes(self) -> tuple[tuple, ...]:
        """
        Return the communicating classes of the finite Markov chain.
        """

        unassigned = set(self.states)
        classes = []

        while unassigned:
            state = next(iter(unassigned))

            class_members = tuple(
                candidate
                for candidate in self.states
                if self.communicates(
                    state,
                    candidate,
                )
            )

            classes.append(class_members)

            for member in class_members:
                unassigned.discard(member)

        return tuple(classes)

    def is_irreducible(self) -> bool:
        """
        Return whether every state communicates with every other state.
        """

        if self.dimension == 0:
            return False

        source = self.states[0]

        return (
            len(
                self.reachable_states(source)
            )
            == self.dimension
            and all(
                self.communicates(
                    source,
                    state,
                )
                for state in self.states
            )
        )

    # -----------------------------------------------------------------------
    # Random Trajectories
    # -----------------------------------------------------------------------

    def sample_trajectory(
        self,
        initial_state,
        *,
        steps: int,
        rng=None,
    ) -> tuple:
        """
        Sample a state trajectory of length ``steps + 1``.

        Parameters
        ----------
        initial_state
            Initial state label.

        steps
            Number of transitions.

        rng
            Optional NumPy random generator or integer seed.
        """

        steps = require_nonnegative_integer(
            steps,
            name="steps",
        )

        current_index = self.state_index(
            initial_state
        )

        if rng is None:
            generator = np.random.default_rng()
        elif isinstance(
            rng,
            np.random.Generator,
        ):
            generator = rng
        else:
            generator = np.random.default_rng(
                rng
            )

        trajectory = [
            self.states[current_index]
        ]

        for _ in range(steps):
            if self.is_row_stochastic:
                probabilities = self.matrix[
                    current_index,
                    :,
                ]
            else:
                probabilities = self.matrix[
                    :,
                    current_index,
                ]

            current_index = int(
                generator.choice(
                    self.dimension,
                    p=probabilities,
                )
            )

            trajectory.append(
                self.states[current_index]
            )

        return tuple(trajectory)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Return structural information about the Markov operator.
        """

        classes = self.communicating_classes()

        return {
            "name": self.name,
            "dimension": self.dimension,
            "states": self.states,
            "convention": self.convention.value,
            "absorbing_states": self.absorbing_states(),
            "communicating_classes": classes,
            "num_communicating_classes": len(classes),
            "irreducible": self.is_irreducible(),
        }
