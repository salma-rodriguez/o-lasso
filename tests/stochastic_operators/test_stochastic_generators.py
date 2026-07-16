"""
Tests for stochastic_operators.generators.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from operator_core import (
    DimensionMismatchError,
    NonSquareOperatorError,
    OperatorError,
)
from stochastic_operators import (
    MarkovGenerator,
    MarkovOperator,
    StochasticConvention,
)


# ===========================================================================
# Construction and Validation
# ===========================================================================

def test_row_markov_generator():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    assert generator.dimension == 2
    assert generator.states == ("a", "b")
    assert generator.is_row_generator
    assert not generator.is_column_generator
    assert generator.convention is StochasticConvention.ROW
    assert generator.is_conservative()


def test_column_markov_generator():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        states=("a", "b"),
        convention="column",
    )

    assert generator.dimension == 2
    assert generator.states == ("a", "b")
    assert generator.is_column_generator
    assert not generator.is_row_generator
    assert generator.convention is StochasticConvention.COLUMN
    assert generator.is_conservative()


def test_default_state_labels():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    assert generator.states == (0, 1)
    assert generator.state_index(0) == 0
    assert generator.state_index(1) == 1
    assert generator.state_label(0) == 0
    assert generator.state_label(1) == 1


def test_generator_matrix_is_read_only():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(ValueError):
        generator.matrix[0, 0] = -1.0


def test_non_square_generator_rejected():
    with pytest.raises(NonSquareOperatorError):
        MarkovGenerator(
            np.ones((2, 3))
        )


def test_non_matrix_generator_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [1.0, -1.0]
        )


def test_complex_generator_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator([
            [-1.0 + 1.0j, 1.0],
            [1.0, -1.0],
        ])


def test_nonfinite_generator_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator([
            [-np.inf, np.inf],
            [0.1, -0.1],
        ])


def test_negative_off_diagonal_rate_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator([
            [-0.3, 0.3],
            [-0.1, 0.1],
        ])


def test_positive_diagonal_rate_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator([
            [0.1, -0.1],
            [0.2, -0.2],
        ])


def test_nonzero_row_sums_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator([
            [-0.3, 0.2],
            [0.1, -0.1],
        ])


def test_nonzero_column_sums_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [
                [-0.3, 0.2],
                [0.3, -0.1],
            ],
            convention="column",
        )


def test_invalid_convention_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [
                [-0.3, 0.3],
                [0.1, -0.1],
            ],
            convention="diagonal",
        )


def test_invalid_tolerance_rejected():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [
                [-0.3, 0.3],
                [0.1, -0.1],
            ],
            tol=-1e-10,
        )


def test_state_labels_must_match_dimension():
    with pytest.raises(DimensionMismatchError):
        MarkovGenerator(
            [
                [-0.3, 0.3],
                [0.1, -0.1],
            ],
            states=("a",),
        )


def test_state_labels_must_be_unique():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [
                [-0.3, 0.3],
                [0.1, -0.1],
            ],
            states=("a", "a"),
        )


def test_metadata_must_be_dictionary():
    with pytest.raises(OperatorError):
        MarkovGenerator(
            [
                [-0.3, 0.3],
                [0.1, -0.1],
            ],
            metadata="invalid",
        )


# ===========================================================================
# State Labels and Rates
# ===========================================================================

def test_state_lookup():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    assert generator.state_index("a") == 0
    assert generator.state_index("b") == 1
    assert generator.state_label(0) == "a"
    assert generator.state_label(1) == "b"


def test_unknown_state_rejected():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    with pytest.raises(OperatorError):
        generator.state_index("missing")


def test_invalid_state_index_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.state_label(2)

    with pytest.raises(OperatorError):
        generator.state_label(-1)

    with pytest.raises(OperatorError):
        generator.state_label(0.5)


def test_row_generator_jump_rates():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    assert np.isclose(
        generator.jump_rate("a", "b"),
        0.3,
    )
    assert np.isclose(
        generator.jump_rate("b", "a"),
        0.1,
    )
    assert np.isclose(
        generator.jump_rate("a", "a"),
        0.0,
    )


def test_column_generator_jump_rates():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        states=("a", "b"),
        convention="column",
    )

    assert np.isclose(
        generator.jump_rate("a", "b"),
        0.3,
    )
    assert np.isclose(
        generator.jump_rate("b", "a"),
        0.1,
    )


def test_exit_and_holding_rates():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    assert np.isclose(
        generator.exit_rate("a"),
        0.3,
    )
    assert np.isclose(
        generator.exit_rate("b"),
        0.1,
    )
    assert np.allclose(
        generator.holding_rates(),
        [0.3, 0.1],
    )


def test_holding_rates_are_read_only():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    rates = generator.holding_rates()

    with pytest.raises(ValueError):
        rates[0] = 1.0


# ===========================================================================
# Transition Semigroup
# ===========================================================================

def test_zero_time_transition_is_identity():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    transition = generator.transition_matrix(
        0.0
    )

    assert np.allclose(
        transition,
        np.eye(2),
    )


def test_transition_matrix_matches_exponential():
    matrix = np.array([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    generator = MarkovGenerator(matrix)

    transition = generator.transition_matrix(
        2.0
    )

    assert np.allclose(
        transition,
        expm(2.0 * matrix),
    )


def test_row_transition_matrix_is_stochastic():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    transition = generator.transition_matrix(
        3.0
    )

    assert np.all(
        transition >= 0.0
    )
    assert np.allclose(
        transition.sum(axis=1),
        np.ones(2),
    )


def test_column_transition_matrix_is_stochastic():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        convention="column",
    )

    transition = generator.transition_matrix(
        3.0
    )

    assert np.all(
        transition >= 0.0
    )
    assert np.allclose(
        transition.sum(axis=0),
        np.ones(2),
    )


def test_transition_matrix_is_read_only():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    transition = generator.transition_matrix(
        1.0
    )

    with pytest.raises(ValueError):
        transition[0, 0] = 0.0


def test_negative_time_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.transition_matrix(-1.0)


def test_nonfinite_time_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.transition_matrix(
            np.inf
        )


def test_nonscalar_time_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.transition_matrix(
            [1.0]
        )


# ===========================================================================
# Transition Operator and Probabilities
# ===========================================================================

def test_transition_operator():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
        name="Q",
    )

    operator = generator.transition_operator(
        2.0
    )

    assert isinstance(
        operator,
        MarkovOperator,
    )
    assert operator.states == ("a", "b")
    assert operator.name == "Q@t=2"
    assert operator.is_row_stochastic
    assert np.allclose(
        operator.matrix,
        generator.transition_matrix(2.0),
    )


def test_named_transition_operator():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    operator = generator.transition_operator(
        1.0,
        name="P_one",
    )

    assert operator.name == "P_one"


def test_row_transition_probability():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    transition = generator.transition_matrix(
        2.0
    )

    assert np.isclose(
        generator.transition_probability(
            "a",
            "b",
            t=2.0,
        ),
        transition[0, 1],
    )


def test_column_transition_probability():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        states=("a", "b"),
        convention="column",
    )

    transition = generator.transition_matrix(
        2.0
    )

    assert np.isclose(
        generator.transition_probability(
            "a",
            "b",
            t=2.0,
        ),
        transition[1, 0],
    )


def test_zero_time_transition_probabilities():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
    )

    assert np.isclose(
        generator.transition_probability(
            "a",
            "a",
            t=0.0,
        ),
        1.0,
    )
    assert np.isclose(
        generator.transition_probability(
            "a",
            "b",
            t=0.0,
        ),
        0.0,
    )


# ===========================================================================
# Distribution Evolution
# ===========================================================================

def test_row_distribution_evolution():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    transition = generator.transition_matrix(
        2.0
    )

    result = generator.evolve_distribution(
        [1.0, 0.0],
        t=2.0,
    )

    assert np.allclose(
        result,
        np.array([1.0, 0.0])
        @ transition,
    )


def test_column_distribution_evolution():
    generator = MarkovGenerator(
        [
            [-0.3, 0.1],
            [0.3, -0.1],
        ],
        convention="column",
    )

    transition = generator.transition_matrix(
        2.0
    )

    result = generator.evolve_distribution(
        [1.0, 0.0],
        t=2.0,
    )

    assert np.allclose(
        result,
        transition
        @ np.array([1.0, 0.0]),
    )


def test_zero_time_distribution_is_unchanged():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    initial = np.array([
        0.4,
        0.6,
    ])

    result = generator.evolve_distribution(
        initial,
        t=0.0,
    )

    assert np.allclose(
        result,
        initial,
    )


def test_evolved_distribution_is_read_only():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    result = generator.evolve_distribution(
        [1.0, 0.0],
        t=1.0,
    )

    with pytest.raises(ValueError):
        result[0] = 0.0


def test_distribution_dimension_mismatch_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(
        DimensionMismatchError
    ):
        generator.evolve_distribution(
            [0.2, 0.3, 0.5],
            t=1.0,
        )


def test_distribution_must_be_one_dimensional():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.evolve_distribution(
            [[1.0, 0.0]],
            t=1.0,
        )


def test_distribution_must_be_nonnegative():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.evolve_distribution(
            [1.1, -0.1],
            t=1.0,
        )


def test_distribution_must_sum_to_one():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.evolve_distribution(
            [0.8, 0.8],
            t=1.0,
        )


# ===========================================================================
# Distribution Histories
# ===========================================================================

def test_distribution_history():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    times = np.array([
        0.0,
        1.0,
        2.0,
    ])

    history = generator.distribution_history(
        [1.0, 0.0],
        times,
    )

    expected = np.vstack([
        generator.evolve_distribution(
            [1.0, 0.0],
            t=time,
        )
        for time in times
    ])

    assert history.shape == (3, 2)
    assert np.allclose(
        history,
        expected,
    )


def test_distribution_history_is_read_only():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    history = generator.distribution_history(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    with pytest.raises(ValueError):
        history[0, 0] = 0.0


def test_empty_times_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.distribution_history(
            [1.0, 0.0],
            [],
        )


def test_multidimensional_times_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.distribution_history(
            [1.0, 0.0],
            [[0.0, 1.0]],
        )


def test_negative_history_time_rejected():
    generator = MarkovGenerator([
        [-0.3, 0.3],
        [0.1, -0.1],
    ])

    with pytest.raises(OperatorError):
        generator.distribution_history(
            [1.0, 0.0],
            [0.0, -1.0],
        )


# ===========================================================================
# Diagnostics and Summary
# ===========================================================================

def test_generator_summary():
    generator = MarkovGenerator(
        [
            [-0.3, 0.3],
            [0.1, -0.1],
        ],
        states=("a", "b"),
        name="Q",
    )

    summary = generator.summary()

    assert summary["name"] == "Q"
    assert summary["dimension"] == 2
    assert summary["states"] == ("a", "b")
    assert summary["convention"] == "row"
    assert summary["conservative"]
    assert np.allclose(
        summary["holding_rates"],
        (0.3, 0.1),
    )
    assert np.isclose(
        summary["max_exit_rate"],
        0.3,
    )
    assert np.isclose(
        summary["min_exit_rate"],
        0.1,
    )
