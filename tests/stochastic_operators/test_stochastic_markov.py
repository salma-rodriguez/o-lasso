"""
Tests for stochastic_operators.markov.
"""

import numpy as np
import pytest

from operator_core import (
    DimensionMismatchError,
    OperatorError,
)
from stochastic_operators import MarkovOperator


# ===========================================================================
# Construction and State Labels
# ===========================================================================

def test_markov_operator_default_states():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    assert operator.states == (0, 1)
    assert operator.state_index(0) == 0
    assert operator.state_index(1) == 1
    assert operator.state_label(0) == 0
    assert operator.state_label(1) == 1


def test_markov_operator_custom_states():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("sunny", "rainy"),
    )

    assert operator.states == (
        "sunny",
        "rainy",
    )
    assert operator.state_index("sunny") == 0
    assert operator.state_index("rainy") == 1
    assert operator.state_label(0) == "sunny"
    assert operator.state_label(1) == "rainy"


def test_state_labels_must_match_dimension():
    with pytest.raises(
        DimensionMismatchError
    ):
        MarkovOperator(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            states=("a",),
        )


def test_state_labels_must_be_unique():
    with pytest.raises(OperatorError):
        MarkovOperator(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ],
            states=("a", "a"),
        )


def test_unknown_state_rejected():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
    )

    with pytest.raises(OperatorError):
        operator.state_index("missing")


def test_invalid_state_index_rejected():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(OperatorError):
        operator.state_label(2)


# ===========================================================================
# Transition Probabilities
# ===========================================================================

def test_one_step_transition_probability():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
    )

    assert np.isclose(
        operator.transition_probability(
            "a",
            "b",
        ),
        0.2,
    )

    assert np.isclose(
        operator.transition_probability(
            "b",
            "a",
        ),
        0.3,
    )


def test_zero_step_transition_probability():
    operator = MarkovOperator(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        states=("a", "b"),
    )

    assert np.isclose(
        operator.transition_probability(
            "a",
            "a",
            steps=0,
        ),
        1.0,
    )

    assert np.isclose(
        operator.transition_probability(
            "a",
            "b",
            steps=0,
        ),
        0.0,
    )


def test_two_step_transition_probability():
    matrix = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    operator = MarkovOperator(
        matrix,
        states=("a", "b"),
    )

    expected = np.linalg.matrix_power(
        matrix,
        2,
    )[0, 1]

    assert np.isclose(
        operator.transition_probability(
            "a",
            "b",
            steps=2,
        ),
        expected,
    )


def test_column_stochastic_transition_probability():
    operator = MarkovOperator(
        [
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        states=("a", "b"),
        convention="column",
    )

    assert np.isclose(
        operator.transition_probability(
            "a",
            "b",
        ),
        0.2,
    )

    assert np.isclose(
        operator.transition_probability(
            "b",
            "a",
        ),
        0.3,
    )


def test_transition_matrix_is_read_only():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    matrix = operator.transition_matrix(
        steps=2
    )

    with pytest.raises(ValueError):
        matrix[0, 0] = 1.0


# ===========================================================================
# Distribution Evolution
# ===========================================================================

def test_evolve_row_distribution():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    result = operator.evolve_distribution(
        [1.0, 0.0],
        steps=1,
    )

    assert np.allclose(
        result,
        [0.8, 0.2],
    )


def test_evolve_column_distribution():
    operator = MarkovOperator(
        [
            [0.8, 0.3],
            [0.2, 0.7],
        ],
        convention="column",
    )

    result = operator.evolve_distribution(
        [1.0, 0.0],
        steps=1,
    )

    assert np.allclose(
        result,
        [0.8, 0.2],
    )


def test_evolve_distribution_zero_steps():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    initial = np.array([
        0.4,
        0.6,
    ])

    result = operator.evolve_distribution(
        initial,
        steps=0,
    )

    assert np.allclose(
        result,
        initial,
    )


def test_distribution_dimension_mismatch_rejected():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(
        DimensionMismatchError
    ):
        operator.evolve_distribution(
            [0.2, 0.3, 0.5]
        )


def test_invalid_distribution_rejected():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    with pytest.raises(OperatorError):
        operator.evolve_distribution(
            [0.8, 0.8]
        )


def test_distribution_history():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    history = operator.distribution_history(
        [1.0, 0.0],
        steps=2,
    )

    expected = np.array([
        [1.0, 0.0],
        [0.8, 0.2],
        [0.70, 0.30],
    ])

    assert history.shape == (3, 2)
    assert np.allclose(
        history,
        expected,
    )


def test_distribution_history_is_read_only():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    history = operator.distribution_history(
        [1.0, 0.0],
        steps=1,
    )

    with pytest.raises(ValueError):
        history[0, 0] = 0.0


# ===========================================================================
# Absorbing States
# ===========================================================================

def test_absorbing_state_detection():
    operator = MarkovOperator(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.5, 0.3],
            [0.0, 0.0, 1.0],
        ],
        states=("left", "middle", "right"),
    )

    assert operator.is_absorbing("left")
    assert not operator.is_absorbing(
        "middle"
    )
    assert operator.is_absorbing("right")

    assert operator.absorbing_states() == (
        "left",
        "right",
    )


def test_column_stochastic_absorbing_state():
    operator = MarkovOperator(
        [
            [1.0, 0.2],
            [0.0, 0.8],
        ],
        states=("fixed", "moving"),
        convention="column",
    )

    assert operator.is_absorbing("fixed")
    assert not operator.is_absorbing(
        "moving"
    )


# ===========================================================================
# Reachability and Communication
# ===========================================================================

def test_reachable_states():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
    )

    assert operator.reachable_states(
        "a"
    ) == (
        "a",
        "b",
        "c",
    )

    assert operator.reachable_states(
        "b",
        include_source=False,
    ) == (
        "c",
    )

    assert operator.reachable_states(
        "c"
    ) == (
        "c",
    )


def test_communication_relation():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
    )

    assert operator.communicates(
        "a",
        "b",
    )
    assert not operator.communicates(
        "a",
        "c",
    )


def test_communicating_classes():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
    )

    classes = operator.communicating_classes()

    assert set(classes) == {
        ("a", "b"),
        ("c",),
    }


def test_irreducible_chain():
    operator = MarkovOperator(
        [
            [0.5, 0.5],
            [0.25, 0.75],
        ],
        states=("a", "b"),
    )

    assert operator.is_irreducible()


def test_reducible_chain():
    operator = MarkovOperator(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ],
        states=("a", "b"),
    )

    assert not operator.is_irreducible()


def test_adjacency_matrix_is_read_only():
    operator = MarkovOperator([
        [0.8, 0.2],
        [0.3, 0.7],
    ])

    adjacency = operator.adjacency_matrix()

    with pytest.raises(ValueError):
        adjacency[0, 0] = False


# ===========================================================================
# Seeded Trajectories
# ===========================================================================

def test_seeded_trajectory_is_reproducible():
    operator = MarkovOperator(
        [
            [0.75, 0.25],
            [0.40, 0.60],
        ],
        states=("a", "b"),
    )

    trajectory = operator.sample_trajectory(
        "a",
        steps=6,
        rng=12345,
    )

    assert trajectory == (
        "a",
        "a",
        "a",
        "b",
        "b",
        "a",
        "a",
    )


def test_zero_step_trajectory():
    operator = MarkovOperator(
        [
            [0.75, 0.25],
            [0.40, 0.60],
        ],
        states=("a", "b"),
    )

    trajectory = operator.sample_trajectory(
        "b",
        steps=0,
        rng=12345,
    )

    assert trajectory == ("b",)


# ===========================================================================
# Summary
# ===========================================================================

def test_markov_summary():
    operator = MarkovOperator(
        [
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        states=("a", "b", "c"),
        name="ExampleChain",
    )

    summary = operator.summary()

    assert summary["name"] == "ExampleChain"
    assert summary["dimension"] == 3
    assert summary["states"] == (
        "a",
        "b",
        "c",
    )
    assert summary["convention"] == "row"
    assert summary["absorbing_states"] == (
        "c",
    )
    assert set(
        summary["communicating_classes"]
    ) == {
        ("a", "b"),
        ("c",),
    }
    assert summary[
        "num_communicating_classes"
    ] == 2
    assert not summary["irreducible"]
