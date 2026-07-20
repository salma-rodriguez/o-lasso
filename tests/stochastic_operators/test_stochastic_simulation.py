"""
Tests for stochastic_operators.monte_carlo.simulation.

This suite covers the primitive simulation layer:

- argument validation;
- state-space validation;
- transition-matrix validation;
- random-generator handling;
- initial-state resolution;
- one-step sampling;
- single-path simulation;
- multiple-path simulation;
- reproducibility;
- deterministic and absorbing chains;
- result metadata and immutability.

Empirical estimators and statistical summaries are tested separately when
those modules are implemented.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from stochastic_operators.monte_carlo import (
    MonteCarloResult,
    simulate_chain,
    simulate_paths,
)
from stochastic_operators.monte_carlo.simulation import (
    _resolve_initial_state,
    _sample_next_state,
    _validate_initial_distribution,
    _validate_initial_state,
    _validate_nonnegative_integer,
    _validate_positive_integer,
    _validate_rng,
    _validate_states,
    _validate_transition_matrix,
)


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def two_states() -> tuple[str, str]:
    return ("a", "b")


@pytest.fixture
def symmetric_matrix() -> np.ndarray:
    return np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=float,
    )


@pytest.fixture
def alternating_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def identity_matrix() -> np.ndarray:
    return np.eye(2, dtype=float)


@pytest.fixture
def absorbing_matrix() -> np.ndarray:
    return np.array(
        [
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=float,
    )


# ============================================================================
# Integer validation
# ============================================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        10,
        np.int32(4),
        np.int64(8),
    ],
)
def test_validate_nonnegative_integer_accepts_valid_values(value) -> None:
    result = _validate_nonnegative_integer(
        value,
        name="steps",
    )

    assert result == int(value)
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        1.5,
        "1",
        None,
        object(),
    ],
)
def test_validate_nonnegative_integer_rejects_invalid_types(value) -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        _validate_nonnegative_integer(
            value,
            name="steps",
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -10,
        np.int64(-5),
    ],
)
def test_validate_nonnegative_integer_rejects_negative_values(value) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_nonnegative_integer(
            value,
            name="steps",
        )


@pytest.mark.parametrize(
    "value",
    [
        1,
        2,
        100,
        np.int64(7),
    ],
)
def test_validate_positive_integer_accepts_valid_values(value) -> None:
    result = _validate_positive_integer(
        value,
        name="n_paths",
    )

    assert result == int(value)


def test_validate_positive_integer_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        _validate_positive_integer(
            0,
            name="n_paths",
        )


def test_validate_positive_integer_rejects_negative_value() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_positive_integer(
            -1,
            name="n_paths",
        )


def test_validate_positive_integer_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        _validate_positive_integer(
            True,
            name="n_paths",
        )


# ============================================================================
# Random-generator validation
# ============================================================================


def test_validate_rng_creates_generator_without_seed() -> None:
    generator, recorded_seed = _validate_rng()

    assert isinstance(generator, np.random.Generator)
    assert recorded_seed is None


def test_validate_rng_creates_generator_with_seed() -> None:
    generator, recorded_seed = _validate_rng(seed=42)

    assert isinstance(generator, np.random.Generator)
    assert recorded_seed == 42


def test_validate_rng_preserves_numpy_integer_seed() -> None:
    generator, recorded_seed = _validate_rng(seed=np.int64(42))

    assert isinstance(generator, np.random.Generator)
    assert recorded_seed == 42
    assert isinstance(recorded_seed, int)


def test_validate_rng_returns_existing_generator() -> None:
    source = np.random.default_rng(10)

    generator, recorded_seed = _validate_rng(rng=source)

    assert generator is source
    assert recorded_seed is None


def test_validate_rng_rejects_seed_and_rng_together() -> None:
    with pytest.raises(
        ValueError,
        match="either seed or rng",
    ):
        _validate_rng(
            seed=42,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize(
    "rng",
    [
        object(),
        np.random.PCG64(),
        "generator",
        42,
    ],
)
def test_validate_rng_rejects_non_generator(rng) -> None:
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        _validate_rng(rng=rng)


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        True,
        1.5,
        "42",
    ],
)
def test_validate_rng_rejects_invalid_seed(seed) -> None:
    with pytest.raises((TypeError, ValueError)):
        _validate_rng(seed=seed)


def test_validate_rng_seed_is_reproducible() -> None:
    left, _ = _validate_rng(seed=123)
    right, _ = _validate_rng(seed=123)

    np.testing.assert_array_equal(
        left.integers(0, 100, size=20),
        right.integers(0, 100, size=20),
    )


# ============================================================================
# State-space validation
# ============================================================================


def test_validate_states_returns_tuple() -> None:
    result = _validate_states(["a", "b", "c"])

    assert result == ("a", "b", "c")
    assert isinstance(result, tuple)


def test_validate_states_preserves_order() -> None:
    result = _validate_states(("c", "a", "b"))

    assert result == ("c", "a", "b")


def test_validate_states_accepts_integer_labels() -> None:
    result = _validate_states([10, 20, 30])

    assert result == (10, 20, 30)


def test_validate_states_accepts_mixed_hashable_labels() -> None:
    result = _validate_states(
        [
            "a",
            1,
            ("tuple", 2),
        ]
    )

    assert result == (
        "a",
        1,
        ("tuple", 2),
    )


def test_validate_states_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _validate_states([])


@pytest.mark.parametrize(
    "states",
    [
        "abc",
        b"abc",
    ],
)
def test_validate_states_rejects_string_like_sequence(states) -> None:
    with pytest.raises(TypeError, match="not a string"):
        _validate_states(states)


@pytest.mark.parametrize(
    "states",
    [
        42,
        None,
        object(),
    ],
)
def test_validate_states_rejects_non_iterable(states) -> None:
    with pytest.raises(TypeError, match="finite sequence"):
        _validate_states(states)


def test_validate_states_rejects_unhashable_label() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _validate_states(
            [
                "a",
                ["b"],
            ]
        )


def test_validate_states_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="unique"):
        _validate_states(
            [
                "a",
                "b",
                "a",
            ]
        )


def test_validate_states_detects_equal_cross_type_duplicates() -> None:
    with pytest.raises(ValueError, match="unique"):
        _validate_states(
            [
                1,
                True,
            ]
        )


# ============================================================================
# Transition-matrix validation
# ============================================================================


def test_validate_transition_matrix_accepts_valid_matrix() -> None:
    source = np.array(
        [
            [0.7, 0.3],
            [0.4, 0.6],
        ]
    )

    result = _validate_transition_matrix(
        source,
        n_states=2,
    )

    np.testing.assert_allclose(result, source)


def test_validate_transition_matrix_converts_to_float64() -> None:
    result = _validate_transition_matrix(
        [
            [1, 0],
            [0, 1],
        ],
        n_states=2,
    )

    assert result.dtype == np.float64


def test_validate_transition_matrix_returns_independent_copy() -> None:
    source = np.eye(2)

    result = _validate_transition_matrix(
        source,
        n_states=2,
    )

    source[0, 0] = 0.0

    np.testing.assert_array_equal(
        result,
        np.eye(2),
    )


def test_validate_transition_matrix_returns_read_only_array() -> None:
    result = _validate_transition_matrix(
        np.eye(2),
        n_states=2,
    )

    assert result.flags.writeable is False

    with pytest.raises(ValueError):
        result[0, 0] = 0.5


@pytest.mark.parametrize(
    "matrix",
    [
        [0.5, 0.5],
        [[1.0]],
        np.ones((2, 3)),
        np.ones((3, 2)),
        np.ones((2, 2, 1)),
    ],
)
def test_validate_transition_matrix_rejects_wrong_shape(matrix) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        _validate_transition_matrix(
            matrix,
            n_states=2,
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [
            [np.nan, 0.0],
            [0.0, 1.0],
        ],
        [
            [np.inf, 0.0],
            [0.0, 1.0],
        ],
        [
            [-np.inf, np.inf],
            [0.0, 1.0],
        ],
    ],
)
def test_validate_transition_matrix_rejects_nonfinite_values(
    matrix,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _validate_transition_matrix(
            matrix,
            n_states=2,
        )


def test_validate_transition_matrix_rejects_negative_probability() -> None:
    with pytest.raises(ValueError, match="negative probabilities"):
        _validate_transition_matrix(
            [
                [1.1, -0.1],
                [0.0, 1.0],
            ],
            n_states=2,
        )


def test_validate_transition_matrix_clips_tiny_negative_roundoff() -> None:
    result = _validate_transition_matrix(
        [
            [1.0 + 1e-14, -1e-14],
            [0.0, 1.0],
        ],
        n_states=2,
    )

    assert result[0, 1] == pytest.approx(0.0)
    assert result[0, 0] == pytest.approx(1.0 + 1e-14)


@pytest.mark.parametrize(
    "matrix",
    [
        [
            [0.8, 0.1],
            [0.5, 0.5],
        ],
        [
            [0.8, 0.3],
            [0.5, 0.5],
        ],
        [
            [0.0, 0.0],
            [0.5, 0.5],
        ],
    ],
)
def test_validate_transition_matrix_rejects_rows_not_summing_to_one(
    matrix,
) -> None:
    with pytest.raises(ValueError, match="row must sum to one"):
        _validate_transition_matrix(
            matrix,
            n_states=2,
        )


def test_validate_transition_matrix_uses_absolute_tolerance() -> None:
    result = _validate_transition_matrix(
        [
            [0.5, 0.5 + 5e-13],
            [0.25, 0.75],
        ],
        n_states=2,
        atol=1e-12,
    )

    assert result.shape == (2, 2)


def test_validate_transition_matrix_rejects_error_beyond_tolerance() -> None:
    with pytest.raises(ValueError, match="row must sum to one"):
        _validate_transition_matrix(
            [
                [0.5, 0.5 + 1e-6],
                [0.25, 0.75],
            ],
            n_states=2,
            atol=1e-12,
        )


def test_validate_transition_matrix_rejects_non_numeric_input() -> None:
    with pytest.raises(TypeError, match="convertible"):
        _validate_transition_matrix(
            [
                ["a", "b"],
                ["c", "d"],
            ],
            n_states=2,
        )


# ============================================================================
# Initial-state validation
# ============================================================================


def test_validate_initial_state_accepts_known_state() -> None:
    mapping = {
        "a": 0,
        "b": 1,
    }

    result = _validate_initial_state(
        "b",
        state_to_index=mapping,
    )

    assert result == "b"


def test_validate_initial_state_rejects_unknown_state() -> None:
    with pytest.raises(ValueError, match="Unknown initial_state"):
        _validate_initial_state(
            "c",
            state_to_index={
                "a": 0,
                "b": 1,
            },
        )


def test_validate_initial_state_rejects_unhashable_state() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _validate_initial_state(
            ["a"],
            state_to_index={
                "a": 0,
                "b": 1,
            },
        )


# ============================================================================
# Initial-distribution validation
# ============================================================================


def test_validate_initial_distribution_accepts_valid_distribution() -> None:
    result = _validate_initial_distribution(
        [0.25, 0.75],
        n_states=2,
    )

    np.testing.assert_allclose(
        result,
        np.array([0.25, 0.75]),
    )


def test_validate_initial_distribution_converts_to_float64() -> None:
    result = _validate_initial_distribution(
        [1, 0],
        n_states=2,
    )

    assert result.dtype == np.float64


def test_validate_initial_distribution_returns_independent_copy() -> None:
    source = np.array([0.2, 0.8])

    result = _validate_initial_distribution(
        source,
        n_states=2,
    )

    source[0] = 1.0

    np.testing.assert_allclose(
        result,
        np.array([0.2, 0.8]),
    )


def test_validate_initial_distribution_returns_read_only_array() -> None:
    result = _validate_initial_distribution(
        [0.2, 0.8],
        n_states=2,
    )

    assert result.flags.writeable is False

    with pytest.raises(ValueError):
        result[0] = 1.0


@pytest.mark.parametrize(
    "distribution",
    [
        [1.0],
        [0.2, 0.3, 0.5],
        [[0.2, 0.8]],
        np.ones((2, 1)),
    ],
)
def test_validate_initial_distribution_rejects_wrong_shape(
    distribution,
) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        _validate_initial_distribution(
            distribution,
            n_states=2,
        )


@pytest.mark.parametrize(
    "distribution",
    [
        [np.nan, 1.0],
        [np.inf, 0.0],
        [-np.inf, np.inf],
    ],
)
def test_validate_initial_distribution_rejects_nonfinite_values(
    distribution,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _validate_initial_distribution(
            distribution,
            n_states=2,
        )


def test_validate_initial_distribution_rejects_negative_probability() -> None:
    with pytest.raises(ValueError, match="negative probabilities"):
        _validate_initial_distribution(
            [1.1, -0.1],
            n_states=2,
        )


def test_validate_initial_distribution_clips_tiny_negative_roundoff() -> None:
    result = _validate_initial_distribution(
        [1.0 + 1e-14, -1e-14],
        n_states=2,
    )

    assert result[0] == pytest.approx(1.0 + 1e-14)
    assert result[1] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "distribution",
    [
        [0.2, 0.2],
        [0.8, 0.8],
        [0.0, 0.0],
    ],
)
def test_validate_initial_distribution_rejects_sum_not_one(
    distribution,
) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _validate_initial_distribution(
            distribution,
            n_states=2,
        )


def test_validate_initial_distribution_rejects_non_numeric_input() -> None:
    with pytest.raises(TypeError, match="convertible"):
        _validate_initial_distribution(
            ["a", "b"],
            n_states=2,
        )


# ============================================================================
# Initial-state resolution
# ============================================================================


def test_resolve_initial_state_returns_fixed_state() -> None:
    generator = np.random.default_rng(42)

    result = _resolve_initial_state(
        states=("a", "b"),
        state_to_index={
            "a": 0,
            "b": 1,
        },
        initial_state="b",
        initial_distribution=None,
        rng=generator,
    )

    assert result == "b"


def test_resolve_initial_state_rejects_both_initialization_methods() -> None:
    with pytest.raises(
        ValueError,
        match="either initial_state or initial_distribution",
    ):
        _resolve_initial_state(
            states=("a", "b"),
            state_to_index={
                "a": 0,
                "b": 1,
            },
            initial_state="a",
            initial_distribution=[1.0, 0.0],
            rng=np.random.default_rng(42),
        )


def test_resolve_initial_state_requires_initialization_method() -> None:
    with pytest.raises(
        ValueError,
        match="Either initial_state or initial_distribution",
    ):
        _resolve_initial_state(
            states=("a", "b"),
            state_to_index={
                "a": 0,
                "b": 1,
            },
            initial_state=None,
            initial_distribution=None,
            rng=np.random.default_rng(42),
        )


def test_resolve_initial_state_samples_degenerate_distribution() -> None:
    result = _resolve_initial_state(
        states=("a", "b"),
        state_to_index={
            "a": 0,
            "b": 1,
        },
        initial_state=None,
        initial_distribution=[0.0, 1.0],
        rng=np.random.default_rng(42),
    )

    assert result == "b"


def test_resolve_initial_state_is_reproducible() -> None:
    left = _resolve_initial_state(
        states=("a", "b", "c"),
        state_to_index={
            "a": 0,
            "b": 1,
            "c": 2,
        },
        initial_state=None,
        initial_distribution=[0.2, 0.3, 0.5],
        rng=np.random.default_rng(123),
    )

    right = _resolve_initial_state(
        states=("a", "b", "c"),
        state_to_index={
            "a": 0,
            "b": 1,
            "c": 2,
        },
        initial_state=None,
        initial_distribution=[0.2, 0.3, 0.5],
        rng=np.random.default_rng(123),
    )

    assert left == right


# ============================================================================
# One-step sampling
# ============================================================================


def test_sample_next_state_follows_deterministic_transition() -> None:
    result = _sample_next_state(
        "a",
        states=("a", "b"),
        state_to_index={
            "a": 0,
            "b": 1,
        },
        transition_matrix=np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        rng=np.random.default_rng(42),
    )

    assert result == "b"


def test_sample_next_state_preserves_absorbing_state() -> None:
    result = _sample_next_state(
        "b",
        states=("a", "b"),
        state_to_index={
            "a": 0,
            "b": 1,
        },
        transition_matrix=np.array(
            [
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
        rng=np.random.default_rng(42),
    )

    assert result == "b"


def test_sample_next_state_is_reproducible() -> None:
    kwargs = {
        "current_state": "a",
        "states": ("a", "b"),
        "state_to_index": {
            "a": 0,
            "b": 1,
        },
        "transition_matrix": np.array(
            [
                [0.25, 0.75],
                [0.60, 0.40],
            ]
        ),
    }

    left = _sample_next_state(
        **kwargs,
        rng=np.random.default_rng(7),
    )
    right = _sample_next_state(
        **kwargs,
        rng=np.random.default_rng(7),
    )

    assert left == right


# ============================================================================
# simulate_chain: basic behavior
# ============================================================================


def test_simulate_chain_returns_monte_carlo_result(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert isinstance(result, MonteCarloResult)


def test_simulate_chain_sets_method(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.method == "simulate_chain"


def test_simulate_chain_path_has_steps_plus_one_states(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=12,
        seed=42,
    )

    assert len(result.path) == 13
    assert result.steps == 12


def test_simulate_chain_path_starts_at_fixed_initial_state(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="b",
        steps=5,
        seed=42,
    )

    assert result.path[0] == "b"


def test_simulate_chain_all_path_values_belong_to_state_space(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=100,
        seed=42,
    )

    assert set(result.path).issubset(set(two_states))


def test_simulate_chain_stores_states(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.states == two_states


def test_simulate_chain_records_single_path_count(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.n_paths == 1


def test_simulate_chain_does_not_store_paths_collection(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.path is not None
    assert result.paths is None


def test_simulate_chain_zero_steps_returns_initial_state_only(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="b",
        steps=0,
        seed=42,
    )

    assert result.path == ("b",)
    assert result.steps == 0


def test_simulate_chain_path_is_immutable(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=3,
        seed=42,
    )

    assert isinstance(result.path, tuple)

    with pytest.raises(TypeError):
        result.path[0] = "b"


# ============================================================================
# simulate_chain: deterministic chains
# ============================================================================


def test_simulate_chain_alternates_deterministically(
    alternating_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        alternating_matrix,
        two_states,
        initial_state="a",
        steps=6,
        seed=42,
    )

    assert result.path == (
        "a",
        "b",
        "a",
        "b",
        "a",
        "b",
        "a",
    )


def test_simulate_chain_identity_matrix_preserves_initial_state(
    identity_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        identity_matrix,
        two_states,
        initial_state="b",
        steps=20,
        seed=42,
    )

    assert result.path == ("b",) * 21


def test_simulate_chain_absorbing_state_remains_absorbed(
    absorbing_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        absorbing_matrix,
        two_states,
        initial_state="b",
        steps=20,
        seed=42,
    )

    assert result.path == ("b",) * 21


def test_simulate_chain_single_state_chain() -> None:
    result = simulate_chain(
        [[1.0]],
        states=("only",),
        initial_state="only",
        steps=10,
        seed=42,
    )

    assert result.path == ("only",) * 11
    assert result.states == ("only",)


# ============================================================================
# simulate_chain: initial distributions
# ============================================================================


def test_simulate_chain_accepts_initial_distribution(
    identity_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        identity_matrix,
        two_states,
        initial_distribution=[0.0, 1.0],
        steps=5,
        seed=42,
    )

    assert result.path == ("b",) * 6


def test_simulate_chain_initial_distribution_is_reproducible(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_distribution=[0.25, 0.75],
        steps=25,
        seed=123,
    )

    right = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_distribution=[0.25, 0.75],
        steps=25,
        seed=123,
    )

    assert left.path == right.path


def test_simulate_chain_rejects_both_initial_state_and_distribution(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="either initial_state or initial_distribution",
    ):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_state="a",
            initial_distribution=[1.0, 0.0],
            steps=5,
            seed=42,
        )


def test_simulate_chain_requires_initialization(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="Either initial_state or initial_distribution",
    ):
        simulate_chain(
            symmetric_matrix,
            two_states,
            steps=5,
            seed=42,
        )


# ============================================================================
# simulate_chain: reproducibility and RNG behavior
# ============================================================================


def test_simulate_chain_same_seed_produces_same_path(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=100,
        seed=42,
    )

    right = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=100,
        seed=42,
    )

    assert left.path == right.path


def test_simulate_chain_records_seed(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.seed == 42


def test_simulate_chain_records_rng_name(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.rng_name == "PCG64"


def test_simulate_chain_existing_rng_is_advanced(
    symmetric_matrix,
    two_states,
) -> None:
    rng = np.random.default_rng(42)

    first = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=20,
        rng=rng,
    )

    second = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=20,
        rng=rng,
    )

    assert first.seed is None
    assert second.seed is None
    assert first.rng_name == "PCG64"
    assert second.rng_name == "PCG64"

    # This confirms generator state is consumed rather than reset.
    assert first.path != second.path


def test_simulate_chain_equivalent_fresh_rngs_are_reproducible(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=50,
        rng=np.random.default_rng(7),
    )

    right = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=50,
        rng=np.random.default_rng(7),
    )

    assert left.path == right.path


def test_simulate_chain_rejects_seed_and_rng_together(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="either seed or rng",
    ):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=5,
            seed=42,
            rng=np.random.default_rng(42),
        )


# ============================================================================
# simulate_chain: metadata
# ============================================================================


def test_simulate_chain_includes_default_metadata(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert result.metadata["simulation"] == (
        "discrete_time_markov_chain"
    )
    assert result.metadata["state_count"] == 2


def test_simulate_chain_accepts_custom_metadata(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
        metadata={
            "experiment": "baseline",
            "author": "OLASSO",
        },
    )

    assert result.metadata["experiment"] == "baseline"
    assert result.metadata["author"] == "OLASSO"


def test_simulate_chain_custom_metadata_is_copied(
    symmetric_matrix,
    two_states,
) -> None:
    metadata = {
        "experiment": "baseline",
    }

    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
        metadata=metadata,
    )

    metadata["experiment"] = "changed"

    assert result.metadata["experiment"] == "baseline"


def test_simulate_chain_result_metadata_is_read_only(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_chain(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        seed=42,
    )

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        "metadata",
        42,
        object(),
    ],
)
def test_simulate_chain_rejects_non_mapping_metadata(
    symmetric_matrix,
    two_states,
    metadata,
) -> None:
    with pytest.raises(TypeError, match="mapping"):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=5,
            seed=42,
            metadata=metadata,
        )


# ============================================================================
# simulate_chain: validation propagation
# ============================================================================


@pytest.mark.parametrize(
    "steps",
    [
        -1,
        True,
        1.5,
        "10",
    ],
)
def test_simulate_chain_rejects_invalid_steps(
    symmetric_matrix,
    two_states,
    steps,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=steps,
            seed=42,
        )


def test_simulate_chain_rejects_duplicate_states(
    symmetric_matrix,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        simulate_chain(
            symmetric_matrix,
            states=("a", "a"),
            initial_state="a",
            steps=5,
            seed=42,
        )


def test_simulate_chain_rejects_matrix_shape_mismatch(
    two_states,
) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        simulate_chain(
            np.eye(3),
            two_states,
            initial_state="a",
            steps=5,
            seed=42,
        )


def test_simulate_chain_rejects_unknown_initial_state(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(ValueError, match="Unknown initial_state"):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_state="c",
            steps=5,
            seed=42,
        )


def test_simulate_chain_rejects_invalid_initial_distribution(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        simulate_chain(
            symmetric_matrix,
            two_states,
            initial_distribution=[0.2, 0.2],
            steps=5,
            seed=42,
        )


# ============================================================================
# simulate_paths: basic behavior
# ============================================================================


def test_simulate_paths_returns_monte_carlo_result(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert isinstance(result, MonteCarloResult)


def test_simulate_paths_sets_method(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.method == "simulate_paths"


def test_simulate_paths_returns_requested_number_of_paths(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=7,
        seed=42,
    )

    assert len(result.paths) == 7
    assert result.n_paths == 7


def test_simulate_paths_each_path_has_steps_plus_one_states(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=12,
        n_paths=5,
        seed=42,
    )

    assert all(
        len(path) == 13
        for path in result.paths
    )


def test_simulate_paths_each_path_starts_at_fixed_initial_state(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="b",
        steps=5,
        n_paths=10,
        seed=42,
    )

    assert all(
        path[0] == "b"
        for path in result.paths
    )


def test_simulate_paths_all_states_belong_to_state_space(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=20,
        n_paths=10,
        seed=42,
    )

    observed = {
        state
        for path in result.paths
        for state in path
    }

    assert observed.issubset(set(two_states))


def test_simulate_paths_stores_states(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.states == two_states


def test_simulate_paths_does_not_store_single_path(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.path is None
    assert result.paths is not None


def test_simulate_paths_zero_steps_returns_initial_states_only(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="b",
        steps=0,
        n_paths=4,
        seed=42,
    )

    assert result.paths == (
        ("b",),
        ("b",),
        ("b",),
        ("b",),
    )


def test_simulate_paths_are_immutable(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=3,
        n_paths=2,
        seed=42,
    )

    assert isinstance(result.paths, tuple)
    assert all(
        isinstance(path, tuple)
        for path in result.paths
    )

    with pytest.raises(TypeError):
        result.paths[0][0] = "b"


# ============================================================================
# simulate_paths: deterministic behavior
# ============================================================================


def test_simulate_paths_alternate_deterministically(
    alternating_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        alternating_matrix,
        two_states,
        initial_state="a",
        steps=4,
        n_paths=3,
        seed=42,
    )

    expected = (
        "a",
        "b",
        "a",
        "b",
        "a",
    )

    assert result.paths == (
        expected,
        expected,
        expected,
    )


def test_simulate_paths_identity_chain_is_constant(
    identity_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        identity_matrix,
        two_states,
        initial_state="b",
        steps=10,
        n_paths=5,
        seed=42,
    )

    assert result.paths == (("b",) * 11,) * 5


def test_simulate_paths_single_state_chain() -> None:
    result = simulate_paths(
        [[1.0]],
        states=("only",),
        initial_state="only",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.paths == (("only",) * 6,) * 3


# ============================================================================
# simulate_paths: initial distributions
# ============================================================================


def test_simulate_paths_accepts_initial_distribution(
    identity_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        identity_matrix,
        two_states,
        initial_distribution=[0.0, 1.0],
        steps=5,
        n_paths=4,
        seed=42,
    )

    assert result.paths == (("b",) * 6,) * 4


def test_simulate_paths_samples_initial_distribution_per_path(
    identity_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        identity_matrix,
        two_states,
        initial_distribution=[0.5, 0.5],
        steps=0,
        n_paths=20,
        seed=42,
    )

    initial_states = tuple(
        path[0]
        for path in result.paths
    )

    assert set(initial_states).issubset({"a", "b"})
    assert len(initial_states) == 20


def test_simulate_paths_initial_distribution_is_reproducible(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_distribution=[0.3, 0.7],
        steps=10,
        n_paths=8,
        seed=99,
    )

    right = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_distribution=[0.3, 0.7],
        steps=10,
        n_paths=8,
        seed=99,
    )

    assert left.paths == right.paths


# ============================================================================
# simulate_paths: reproducibility and RNG behavior
# ============================================================================


def test_simulate_paths_same_seed_produces_same_paths(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=20,
        n_paths=10,
        seed=42,
    )

    right = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=20,
        n_paths=10,
        seed=42,
    )

    assert left.paths == right.paths


def test_simulate_paths_records_seed(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.seed == 42


def test_simulate_paths_records_rng_name(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.rng_name == "PCG64"


def test_simulate_paths_with_equivalent_rngs_are_reproducible(
    symmetric_matrix,
    two_states,
) -> None:
    left = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=10,
        n_paths=4,
        rng=np.random.default_rng(123),
    )

    right = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=10,
        n_paths=4,
        rng=np.random.default_rng(123),
    )

    assert left.paths == right.paths


def test_simulate_paths_rejects_seed_and_rng_together(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="either seed or rng",
    ):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=5,
            n_paths=3,
            seed=42,
            rng=np.random.default_rng(42),
        )


# ============================================================================
# simulate_paths: metadata
# ============================================================================


def test_simulate_paths_includes_default_metadata(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    assert result.metadata["simulation"] == (
        "discrete_time_markov_chain"
    )
    assert result.metadata["state_count"] == 2
    assert result.metadata["independent_paths"] is True


def test_simulate_paths_accepts_custom_metadata(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
        metadata={
            "experiment": "ensemble",
        },
    )

    assert result.metadata["experiment"] == "ensemble"


def test_simulate_paths_custom_metadata_is_copied(
    symmetric_matrix,
    two_states,
) -> None:
    metadata = {
        "experiment": "ensemble",
    }

    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
        metadata=metadata,
    )

    metadata["experiment"] = "changed"

    assert result.metadata["experiment"] == "ensemble"


def test_simulate_paths_result_metadata_is_read_only(
    symmetric_matrix,
    two_states,
) -> None:
    result = simulate_paths(
        symmetric_matrix,
        two_states,
        initial_state="a",
        steps=5,
        n_paths=3,
        seed=42,
    )

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


# ============================================================================
# simulate_paths: validation propagation
# ============================================================================


@pytest.mark.parametrize(
    "n_paths",
    [
        0,
        -1,
        True,
        1.5,
        "3",
    ],
)
def test_simulate_paths_rejects_invalid_n_paths(
    symmetric_matrix,
    two_states,
    n_paths,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=5,
            n_paths=n_paths,
            seed=42,
        )


@pytest.mark.parametrize(
    "steps",
    [
        -1,
        True,
        1.5,
        "5",
    ],
)
def test_simulate_paths_rejects_invalid_steps(
    symmetric_matrix,
    two_states,
    steps,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=steps,
            n_paths=3,
            seed=42,
        )


def test_simulate_paths_rejects_both_initialization_methods(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="either initial_state or initial_distribution",
    ):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="a",
            initial_distribution=[1.0, 0.0],
            steps=5,
            n_paths=3,
            seed=42,
        )


def test_simulate_paths_requires_initialization(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(
        ValueError,
        match="Either initial_state or initial_distribution",
    ):
        simulate_paths(
            symmetric_matrix,
            two_states,
            steps=5,
            n_paths=3,
            seed=42,
        )


def test_simulate_paths_rejects_unknown_initial_state(
    symmetric_matrix,
    two_states,
) -> None:
    with pytest.raises(ValueError, match="Unknown initial_state"):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="c",
            steps=5,
            n_paths=3,
            seed=42,
        )


def test_simulate_paths_rejects_invalid_matrix(
    two_states,
) -> None:
    with pytest.raises(ValueError, match="row must sum to one"):
        simulate_paths(
            [
                [0.2, 0.2],
                [0.5, 0.5],
            ],
            two_states,
            initial_state="a",
            steps=5,
            n_paths=3,
            seed=42,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        "metadata",
        42,
        object(),
    ],
)
def test_simulate_paths_rejects_non_mapping_metadata(
    symmetric_matrix,
    two_states,
    metadata,
) -> None:
    with pytest.raises(TypeError, match="mapping"):
        simulate_paths(
            symmetric_matrix,
            two_states,
            initial_state="a",
            steps=5,
            n_paths=3,
            seed=42,
            metadata=metadata,
        )


# ============================================================================
# Public API behavior
# ============================================================================


def test_public_simulate_chain_import_is_callable() -> None:
    assert callable(simulate_chain)


def test_public_simulate_paths_import_is_callable() -> None:
    assert callable(simulate_paths)


def test_single_and_multiple_simulation_agree_for_deterministic_chain(
    alternating_matrix,
    two_states,
) -> None:
    single = simulate_chain(
        alternating_matrix,
        two_states,
        initial_state="a",
        steps=6,
        seed=42,
    )

    multiple = simulate_paths(
        alternating_matrix,
        two_states,
        initial_state="a",
        steps=6,
        n_paths=1,
        seed=42,
    )

    assert multiple.paths == (single.path,)
