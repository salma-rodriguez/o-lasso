"""
Tests for stochastic_operators.monte_carlo.

This suite currently covers the immutable MonteCarloResult layer and its
supporting validation/freezing helpers. Simulation and empirical-estimation
tests will be added as those phases are implemented.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from stochastic_operators.monte_carlo.result import (
    MonteCarloResult,
    _freeze_array,
    _freeze_metadata,
    _freeze_path,
    _freeze_paths,
    _validate_confidence_interval,
    _validate_confidence_level,
    _validate_optional_finite_float,
    _validate_optional_nonnegative_integer,
    _validate_optional_positive_integer,
    _validate_seed,
    _values_equal,
)


# ============================================================================
# Helpers
# ============================================================================


def assert_read_only(array: np.ndarray) -> None:
    """Assert that a NumPy array cannot be modified."""

    assert isinstance(array, np.ndarray)
    assert array.flags.writeable is False

    with pytest.raises(ValueError):
        array.flat[0] = 100.0


# ============================================================================
# _freeze_array
# ============================================================================


def test_freeze_array_returns_none_for_none() -> None:
    assert _freeze_array(None, name="value") is None


def test_freeze_array_converts_sequence_to_array() -> None:
    result = _freeze_array(
        [1.0, 2.0, 3.0],
        name="value",
    )

    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0, 3.0]),
    )


def test_freeze_array_applies_requested_dtype() -> None:
    result = _freeze_array(
        [1, 2, 3],
        name="value",
        dtype=float,
    )

    assert result.dtype == np.dtype(float)


def test_freeze_array_returns_independent_copy() -> None:
    source = np.array([1.0, 2.0, 3.0])
    result = _freeze_array(source, name="value")

    source[0] = 99.0

    np.testing.assert_array_equal(
        result,
        np.array([1.0, 2.0, 3.0]),
    )


def test_freeze_array_is_read_only() -> None:
    result = _freeze_array(
        [1.0, 2.0],
        name="value",
    )

    assert result is not None
    assert_read_only(result)


def test_freeze_array_validates_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="exactly 1 dimension",
    ):
        _freeze_array(
            [[1.0, 2.0]],
            name="samples",
            ndim=1,
        )


def test_freeze_array_allows_empty_by_default() -> None:
    result = _freeze_array(
        [],
        name="value",
    )

    assert result is not None
    assert result.size == 0


def test_freeze_array_can_reject_empty_array() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _freeze_array(
            [],
            name="value",
            allow_empty=False,
        )


def test_freeze_array_rejects_unconvertible_value() -> None:
    with pytest.raises(TypeError, match="convertible"):
        _freeze_array(
            object(),
            name="value",
            dtype=float,
        )


# ============================================================================
# _freeze_metadata
# ============================================================================


def test_freeze_metadata_returns_empty_mapping_for_none() -> None:
    result = _freeze_metadata(None)

    assert isinstance(result, MappingProxyType)
    assert dict(result) == {}


def test_freeze_metadata_copies_mapping() -> None:
    source = {"purpose": "testing"}
    result = _freeze_metadata(source)

    source["purpose"] = "changed"

    assert result["purpose"] == "testing"


def test_freeze_metadata_returns_read_only_mapping() -> None:
    result = _freeze_metadata({"a": 1})

    with pytest.raises(TypeError):
        result["b"] = 2


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        (),
        "metadata",
        1,
        object(),
    ],
)
def test_freeze_metadata_rejects_non_mapping(metadata) -> None:
    with pytest.raises(TypeError, match="mapping"):
        _freeze_metadata(metadata)


# ============================================================================
# _freeze_path and _freeze_paths
# ============================================================================


def test_freeze_path_converts_sequence_to_tuple() -> None:
    result = _freeze_path(
        ["a", "b", "c"],
        name="path",
    )

    assert result == ("a", "b", "c")


def test_freeze_path_allows_empty_path_by_default() -> None:
    assert _freeze_path([], name="path") == ()


def test_freeze_path_can_reject_empty_path() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        _freeze_path(
            [],
            name="states",
            allow_empty=False,
        )


@pytest.mark.parametrize("path", ["abc", b"abc"])
def test_freeze_path_rejects_string_like_sequences(path) -> None:
    with pytest.raises(TypeError, match="not a string"):
        _freeze_path(path, name="path")


def test_freeze_path_rejects_non_iterable() -> None:
    with pytest.raises(TypeError, match="finite sequence"):
        _freeze_path(42, name="path")


def test_freeze_path_rejects_unhashable_state() -> None:
    with pytest.raises(TypeError, match="hashable"):
        _freeze_path(
            ("a", ["unhashable"]),
            name="path",
        )


def test_freeze_paths_returns_none_for_none() -> None:
    assert _freeze_paths(None) is None


def test_freeze_paths_converts_nested_sequences() -> None:
    result = _freeze_paths(
        [
            ["a", "b"],
            ["b", "c"],
        ]
    )

    assert result == (
        ("a", "b"),
        ("b", "c"),
    )


def test_freeze_paths_allows_empty_collection() -> None:
    assert _freeze_paths([]) == ()


def test_freeze_paths_allows_empty_individual_paths() -> None:
    result = _freeze_paths(
        [
            [],
            ["a"],
        ]
    )

    assert result == (
        (),
        ("a",),
    )


@pytest.mark.parametrize("paths", ["abc", b"abc"])
def test_freeze_paths_rejects_string_like_value(paths) -> None:
    with pytest.raises(TypeError, match="sequence of state paths"):
        _freeze_paths(paths)


def test_freeze_paths_rejects_non_iterable() -> None:
    with pytest.raises(TypeError, match="finite sequence"):
        _freeze_paths(42)


# ============================================================================
# Integer validation helpers
# ============================================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        10,
        np.int64(7),
    ],
)
def test_validate_optional_nonnegative_integer_accepts_valid_values(
    value,
) -> None:
    result = _validate_optional_nonnegative_integer(
        value,
        name="count",
    )

    assert result == int(value)
    assert isinstance(result, int)


def test_validate_optional_nonnegative_integer_accepts_none() -> None:
    assert (
        _validate_optional_nonnegative_integer(
            None,
            name="count",
        )
        is None
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        1.5,
        "1",
        object(),
    ],
)
def test_validate_optional_nonnegative_integer_rejects_nonintegers(
    value,
) -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        _validate_optional_nonnegative_integer(
            value,
            name="count",
        )


@pytest.mark.parametrize("value", [-1, -10, np.int64(-2)])
def test_validate_optional_nonnegative_integer_rejects_negative_values(
    value,
) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_optional_nonnegative_integer(
            value,
            name="count",
        )


@pytest.mark.parametrize("value", [1, 2, np.int64(5)])
def test_validate_optional_positive_integer_accepts_positive_values(
    value,
) -> None:
    assert (
        _validate_optional_positive_integer(
            value,
            name="thinning",
        )
        == int(value)
    )


def test_validate_optional_positive_integer_accepts_none() -> None:
    assert (
        _validate_optional_positive_integer(
            None,
            name="thinning",
        )
        is None
    )


def test_validate_optional_positive_integer_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        _validate_optional_positive_integer(
            0,
            name="thinning",
        )


# ============================================================================
# Floating-point validation helpers
# ============================================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        1.5,
        np.float64(2.5),
        np.int64(3),
    ],
)
def test_validate_optional_finite_float_accepts_real_values(
    value,
) -> None:
    result = _validate_optional_finite_float(
        value,
        name="value",
    )

    assert result == pytest.approx(float(value))
    assert isinstance(result, float)


def test_validate_optional_finite_float_accepts_none() -> None:
    assert (
        _validate_optional_finite_float(
            None,
            name="value",
        )
        is None
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        "invalid",
        object(),
    ],
)
def test_validate_optional_finite_float_rejects_invalid_types(
    value,
) -> None:
    with pytest.raises(TypeError, match="finite real"):
        _validate_optional_finite_float(
            value,
            name="value",
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_validate_optional_finite_float_rejects_nonfinite_values(
    value,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _validate_optional_finite_float(
            value,
            name="value",
        )


def test_validate_optional_finite_float_can_require_nonnegative() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_optional_finite_float(
            -0.1,
            name="variance",
            nonnegative=True,
        )


def test_validate_optional_finite_float_accepts_zero_when_nonnegative() -> None:
    assert (
        _validate_optional_finite_float(
            0.0,
            name="variance",
            nonnegative=True,
        )
        == pytest.approx(0.0)
    )


# ============================================================================
# Confidence validation
# ============================================================================


@pytest.mark.parametrize(
    "level",
    [
        0.01,
        0.5,
        0.90,
        0.95,
        0.999,
    ],
)
def test_validate_confidence_level_accepts_valid_values(
    level: float,
) -> None:
    assert _validate_confidence_level(level) == pytest.approx(level)


def test_validate_confidence_level_accepts_none() -> None:
    assert _validate_confidence_level(None) is None


@pytest.mark.parametrize(
    "level",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
    ],
)
def test_validate_confidence_level_rejects_out_of_range_values(
    level: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        _validate_confidence_level(level)


@pytest.mark.parametrize(
    "interval",
    [
        (0.1, 0.9),
        [-1.0, 1.0],
        np.array([2.0, 2.0]),
    ],
)
def test_validate_confidence_interval_accepts_valid_intervals(
    interval,
) -> None:
    result = _validate_confidence_interval(interval)

    assert result == pytest.approx(
        (float(interval[0]), float(interval[1]))
    )


def test_validate_confidence_interval_accepts_none() -> None:
    assert _validate_confidence_interval(None) is None


@pytest.mark.parametrize("interval", ["0,1", b"0,1"])
def test_validate_confidence_interval_rejects_string_like_value(
    interval,
) -> None:
    with pytest.raises(TypeError, match="two-element"):
        _validate_confidence_interval(interval)


@pytest.mark.parametrize(
    "interval",
    [
        (),
        (1.0,),
        (1.0, 2.0, 3.0),
    ],
)
def test_validate_confidence_interval_requires_two_endpoints(
    interval,
) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        _validate_confidence_interval(interval)


def test_validate_confidence_interval_rejects_reversed_endpoints() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_confidence_interval((2.0, 1.0))


@pytest.mark.parametrize(
    "interval",
    [
        (np.nan, 1.0),
        (0.0, np.inf),
        (-np.inf, 1.0),
    ],
)
def test_validate_confidence_interval_rejects_nonfinite_endpoints(
    interval,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _validate_confidence_interval(interval)


# ============================================================================
# Seed validation
# ============================================================================


@pytest.mark.parametrize("seed", [0, 1, 100, np.int64(42)])
def test_validate_seed_accepts_nonnegative_integer(seed) -> None:
    assert _validate_seed(seed) == int(seed)


def test_validate_seed_accepts_none() -> None:
    assert _validate_seed(None) is None


@pytest.mark.parametrize(
    "seed",
    [
        -1,
        True,
        1.5,
        "42",
    ],
)
def test_validate_seed_rejects_invalid_seed(seed) -> None:
    with pytest.raises((TypeError, ValueError)):
        _validate_seed(seed)


# ============================================================================
# _values_equal
# ============================================================================


def test_values_equal_handles_equal_scalars() -> None:
    assert _values_equal(1.0, 1.0) is True


def test_values_equal_handles_unequal_scalars() -> None:
    assert _values_equal(1.0, 2.0) is False


def test_values_equal_handles_equal_arrays() -> None:
    assert (
        _values_equal(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        )
        is True
    )


def test_values_equal_handles_unequal_arrays() -> None:
    assert (
        _values_equal(
            np.array([1.0, 2.0]),
            np.array([1.0, 3.0]),
        )
        is False
    )


def test_values_equal_treats_array_nan_values_as_equal() -> None:
    assert (
        _values_equal(
            np.array([1.0, np.nan]),
            np.array([1.0, np.nan]),
        )
        is True
    )


def test_values_equal_handles_array_and_sequence() -> None:
    assert (
        _values_equal(
            np.array([1, 2]),
            [1, 2],
        )
        is True
    )


# ============================================================================
# Minimal MonteCarloResult construction
# ============================================================================


def test_minimal_result_construction() -> None:
    result = MonteCarloResult(method="simulate")

    assert result.method == "simulate"
    assert result.estimate is None
    assert result.samples is None
    assert result.path is None
    assert result.paths is None
    assert result.states is None
    assert result.variance is None
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert result.confidence_level is None
    assert result.n_samples is None
    assert result.n_paths is None
    assert result.steps is None
    assert result.burn_in is None
    assert result.thinning is None
    assert result.seed is None
    assert result.rng_name is None
    assert result.converged is None
    assert dict(result.metadata) == {}


def test_method_is_stripped() -> None:
    result = MonteCarloResult(method="  simulate_chain  ")

    assert result.method == "simulate_chain"


def test_result_accepts_complete_scalar_estimate() -> None:
    result = MonteCarloResult(
        method="estimate_probability",
        estimate=0.42,
        variance=0.01,
        standard_error=0.1,
        confidence_interval=(0.22, 0.62),
        confidence_level=0.95,
        n_samples=100,
        seed=42,
        rng_name="PCG64",
        converged=True,
        metadata={"target": "a"},
    )

    assert result.estimate == pytest.approx(0.42)
    assert result.variance == pytest.approx(0.01)
    assert result.standard_error == pytest.approx(0.1)
    assert result.confidence_interval == pytest.approx((0.22, 0.62))
    assert result.confidence_level == pytest.approx(0.95)
    assert result.n_samples == 100
    assert result.seed == 42
    assert result.rng_name == "PCG64"
    assert result.converged is True
    assert result.metadata["target"] == "a"


def test_result_accepts_vector_estimate() -> None:
    result = MonteCarloResult(
        method="empirical_distribution",
        estimate=np.array([0.25, 0.75]),
        states=("a", "b"),
    )

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.25, 0.75]),
    )


def test_result_accepts_matrix_estimate() -> None:
    estimate = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.75],
        ]
    )

    result = MonteCarloResult(
        method="empirical_transition_matrix",
        estimate=estimate,
        states=("a", "b"),
    )

    np.testing.assert_allclose(result.estimate, estimate)


def test_result_accepts_mapping_estimate() -> None:
    result = MonteCarloResult(
        method="empirical_distribution",
        estimate={
            "a": 0.25,
            "b": 0.75,
        },
    )

    assert isinstance(result.estimate, MappingProxyType)
    assert dict(result.estimate) == {
        "a": 0.25,
        "b": 0.75,
    }


# ============================================================================
# Method validation
# ============================================================================


@pytest.mark.parametrize(
    "method",
    [
        None,
        1,
        object(),
        ["simulate"],
    ],
)
def test_result_rejects_non_string_method(method) -> None:
    with pytest.raises(TypeError, match="method must be a string"):
        MonteCarloResult(method=method)


@pytest.mark.parametrize("method", ["", " ", "\n\t"])
def test_result_rejects_empty_method(method: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        MonteCarloResult(method=method)


# ============================================================================
# Estimate freezing
# ============================================================================


def test_array_estimate_is_copied() -> None:
    source = np.array([0.2, 0.8])

    result = MonteCarloResult(
        method="estimate",
        estimate=source,
    )

    source[0] = 1.0

    np.testing.assert_allclose(
        result.estimate,
        np.array([0.2, 0.8]),
    )


def test_array_estimate_is_read_only() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.2, 0.8]),
    )

    assert_read_only(result.estimate)


def test_list_estimate_is_recursively_converted_to_tuple() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate=[
            [1, 2],
            [3, 4],
        ],
    )

    assert result.estimate == (
        (1, 2),
        (3, 4),
    )


def test_mapping_estimate_is_read_only() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate={"a": 1.0},
    )

    with pytest.raises(TypeError):
        result.estimate["b"] = 2.0


def test_array_nested_in_mapping_estimate_is_read_only() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate={
            "distribution": np.array([0.3, 0.7]),
        },
    )

    assert_read_only(result.estimate["distribution"])


# ============================================================================
# Samples
# ============================================================================


def test_samples_are_converted_to_read_only_array() -> None:
    result = MonteCarloResult(
        method="sample",
        samples=[1.0, 2.0, 3.0],
        n_samples=3,
    )

    np.testing.assert_allclose(
        result.samples,
        np.array([1.0, 2.0, 3.0]),
    )
    assert_read_only(result.samples)


def test_samples_are_copied() -> None:
    source = np.array([1.0, 2.0])

    result = MonteCarloResult(
        method="sample",
        samples=source,
        n_samples=2,
    )

    source[0] = 99.0

    assert result.samples[0] == pytest.approx(1.0)


def test_samples_must_be_one_dimensional() -> None:
    with pytest.raises(ValueError, match="exactly 1 dimension"):
        MonteCarloResult(
            method="sample",
            samples=[[1.0, 2.0]],
        )


def test_empty_samples_are_allowed() -> None:
    result = MonteCarloResult(
        method="sample",
        samples=[],
        n_samples=0,
    )

    assert result.samples.size == 0


def test_n_samples_must_match_samples_size() -> None:
    with pytest.raises(ValueError, match="samples.size"):
        MonteCarloResult(
            method="sample",
            samples=[1.0, 2.0],
            n_samples=3,
        )


# ============================================================================
# Paths and states
# ============================================================================


def test_single_path_is_frozen() -> None:
    source = ["a", "b", "c"]

    result = MonteCarloResult(
        method="simulate_chain",
        path=source,
        steps=2,
        n_paths=1,
    )

    source.append("d")

    assert result.path == ("a", "b", "c")


def test_multiple_paths_are_frozen() -> None:
    result = MonteCarloResult(
        method="simulate_paths",
        paths=[
            ["a", "b"],
            ["b", "a"],
        ],
        steps=1,
        n_paths=2,
    )

    assert result.paths == (
        ("a", "b"),
        ("b", "a"),
    )


def test_states_are_frozen() -> None:
    source = ["a", "b"]

    result = MonteCarloResult(
        method="estimate",
        states=source,
    )

    source.append("c")

    assert result.states == ("a", "b")


def test_states_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        MonteCarloResult(
            method="estimate",
            states=(),
        )


def test_states_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        MonteCarloResult(
            method="estimate",
            states=("a", "a"),
        )


def test_states_must_be_hashable() -> None:
    with pytest.raises(TypeError, match="hashable"):
        MonteCarloResult(
            method="estimate",
            states=("a", ["b"]),
        )


def test_result_rejects_path_and_paths_together() -> None:
    with pytest.raises(
        ValueError,
        match="either path or paths",
    ):
        MonteCarloResult(
            method="simulate",
            path=("a",),
            paths=(("a",),),
        )


def test_steps_must_equal_single_path_transition_count() -> None:
    with pytest.raises(ValueError, match=r"len\(path\) - 1"):
        MonteCarloResult(
            method="simulate_chain",
            path=("a", "b", "c"),
            steps=3,
        )


def test_zero_step_path_contains_initial_state() -> None:
    result = MonteCarloResult(
        method="simulate_chain",
        path=("a",),
        steps=0,
        n_paths=1,
    )

    assert result.steps == 0
    assert result.path == ("a",)


def test_n_paths_must_be_one_for_single_path() -> None:
    with pytest.raises(ValueError, match="equal one"):
        MonteCarloResult(
            method="simulate_chain",
            path=("a", "b"),
            steps=1,
            n_paths=2,
        )


def test_n_paths_must_match_number_of_paths() -> None:
    with pytest.raises(ValueError, match=r"len\(paths\)"):
        MonteCarloResult(
            method="simulate_paths",
            paths=(
                ("a", "b"),
                ("b", "a"),
            ),
            steps=1,
            n_paths=3,
        )


def test_each_path_must_have_steps_plus_one_states() -> None:
    with pytest.raises(ValueError, match=r"steps \+ 1"):
        MonteCarloResult(
            method="simulate_paths",
            paths=(
                ("a", "b"),
                ("a", "b", "c"),
            ),
            steps=1,
            n_paths=2,
        )


def test_empty_paths_collection_can_have_zero_paths() -> None:
    result = MonteCarloResult(
        method="simulate_paths",
        paths=(),
        n_paths=0,
    )

    assert result.paths == ()
    assert result.n_paths == 0


# ============================================================================
# Variance and standard error
# ============================================================================


@pytest.mark.parametrize(
    "variance",
    [
        0.0,
        0.25,
        np.float64(1.5),
    ],
)
def test_result_accepts_nonnegative_scalar_variance(
    variance,
) -> None:
    result = MonteCarloResult(
        method="estimate",
        variance=variance,
    )

    assert result.variance == pytest.approx(float(variance))


@pytest.mark.parametrize(
    "standard_error",
    [
        0.0,
        0.1,
        np.float64(0.25),
    ],
)
def test_result_accepts_nonnegative_scalar_standard_error(
    standard_error,
) -> None:
    result = MonteCarloResult(
        method="estimate",
        standard_error=standard_error,
    )

    assert result.standard_error == pytest.approx(
        float(standard_error)
    )


def test_result_accepts_array_variance() -> None:
    result = MonteCarloResult(
        method="estimate",
        variance=np.array([0.1, 0.2]),
    )

    np.testing.assert_allclose(
        result.variance,
        np.array([0.1, 0.2]),
    )
    assert_read_only(result.variance)


def test_result_accepts_array_standard_error() -> None:
    result = MonteCarloResult(
        method="estimate",
        standard_error=np.array([0.01, 0.02]),
    )

    np.testing.assert_allclose(
        result.standard_error,
        np.array([0.01, 0.02]),
    )
    assert_read_only(result.standard_error)


@pytest.mark.parametrize("value", [-0.1, -1.0])
def test_result_rejects_negative_scalar_variance(value: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MonteCarloResult(
            method="estimate",
            variance=value,
        )


@pytest.mark.parametrize("value", [-0.1, -1.0])
def test_result_rejects_negative_scalar_standard_error(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MonteCarloResult(
            method="estimate",
            standard_error=value,
        )


def test_result_rejects_negative_array_variance() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MonteCarloResult(
            method="estimate",
            variance=np.array([0.1, -0.1]),
        )


def test_result_rejects_negative_array_standard_error() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MonteCarloResult(
            method="estimate",
            standard_error=np.array([0.1, -0.1]),
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_result_rejects_nonfinite_scalar_variance(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        MonteCarloResult(
            method="estimate",
            variance=value,
        )


def test_result_rejects_nonfinite_array_standard_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        MonteCarloResult(
            method="estimate",
            standard_error=np.array([0.1, np.nan]),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "variance",
        "standard_error",
    ],
)
def test_result_rejects_boolean_uncertainty_statistics(
    field_name: str,
) -> None:
    with pytest.raises(TypeError, match="nonnegative real"):
        MonteCarloResult(
            method="estimate",
            **{field_name: True},
        )


# ============================================================================
# Confidence interval consistency
# ============================================================================


def test_result_accepts_confidence_interval_and_level() -> None:
    result = MonteCarloResult(
        method="estimate",
        confidence_interval=(0.2, 0.8),
        confidence_level=0.95,
    )

    assert result.confidence_interval == pytest.approx((0.2, 0.8))
    assert result.confidence_level == pytest.approx(0.95)


def test_result_requires_level_when_interval_is_provided() -> None:
    with pytest.raises(
        ValueError,
        match="confidence_level is required",
    ):
        MonteCarloResult(
            method="estimate",
            confidence_interval=(0.2, 0.8),
        )


def test_result_requires_interval_when_level_is_provided() -> None:
    with pytest.raises(
        ValueError,
        match="confidence_interval is required",
    ):
        MonteCarloResult(
            method="estimate",
            confidence_level=0.95,
        )


# ============================================================================
# Count and bookkeeping validation
# ============================================================================


@pytest.mark.parametrize(
    "field_name",
    [
        "n_samples",
        "n_paths",
        "steps",
        "burn_in",
    ],
)
def test_result_accepts_zero_for_nonnegative_counts(
    field_name: str,
) -> None:
    result = MonteCarloResult(
        method="estimate",
        **{field_name: 0},
    )

    assert getattr(result, field_name) == 0


@pytest.mark.parametrize(
    "field_name",
    [
        "n_samples",
        "n_paths",
        "steps",
        "burn_in",
    ],
)
def test_result_rejects_negative_counts(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        MonteCarloResult(
            method="estimate",
            **{field_name: -1},
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "n_samples",
        "n_paths",
        "steps",
        "burn_in",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        1.5,
        "1",
    ],
)
def test_result_rejects_noninteger_counts(
    field_name: str,
    value,
) -> None:
    with pytest.raises(TypeError, match="nonnegative integer"):
        MonteCarloResult(
            method="estimate",
            **{field_name: value},
        )


@pytest.mark.parametrize("thinning", [1, 2, 10])
def test_result_accepts_positive_thinning(thinning: int) -> None:
    result = MonteCarloResult(
        method="estimate",
        thinning=thinning,
    )

    assert result.thinning == thinning


@pytest.mark.parametrize("thinning", [0, -1])
def test_result_rejects_nonpositive_thinning(thinning: int) -> None:
    with pytest.raises(ValueError):
        MonteCarloResult(
            method="estimate",
            thinning=thinning,
        )


def test_burn_in_cannot_exceed_steps() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        MonteCarloResult(
            method="estimate",
            steps=10,
            burn_in=11,
        )


def test_burn_in_may_equal_steps() -> None:
    result = MonteCarloResult(
        method="estimate",
        steps=10,
        burn_in=10,
    )

    assert result.burn_in == 10


# ============================================================================
# Seed, RNG, and convergence fields
# ============================================================================


def test_result_accepts_seed_zero() -> None:
    result = MonteCarloResult(
        method="simulate",
        seed=0,
    )

    assert result.seed == 0


def test_rng_name_is_stripped() -> None:
    result = MonteCarloResult(
        method="simulate",
        rng_name="  PCG64  ",
    )

    assert result.rng_name == "PCG64"


@pytest.mark.parametrize("rng_name", ["", " ", "\n"])
def test_result_rejects_empty_rng_name(rng_name: str) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        MonteCarloResult(
            method="simulate",
            rng_name=rng_name,
        )


@pytest.mark.parametrize("rng_name", [1, object(), []])
def test_result_rejects_non_string_rng_name(rng_name) -> None:
    with pytest.raises(TypeError, match="string or None"):
        MonteCarloResult(
            method="simulate",
            rng_name=rng_name,
        )


@pytest.mark.parametrize("converged", [True, False, np.bool_(True)])
def test_result_accepts_boolean_convergence(converged) -> None:
    result = MonteCarloResult(
        method="estimate",
        converged=converged,
    )

    assert isinstance(result.converged, bool)
    assert result.converged is bool(converged)


@pytest.mark.parametrize("converged", [0, 1, "yes"])
def test_result_rejects_non_boolean_convergence(converged) -> None:
    with pytest.raises(TypeError, match="boolean or None"):
        MonteCarloResult(
            method="estimate",
            converged=converged,
        )


# ============================================================================
# Metadata
# ============================================================================


def test_result_metadata_is_copied() -> None:
    metadata = {"experiment": "baseline"}

    result = MonteCarloResult(
        method="estimate",
        metadata=metadata,
    )

    metadata["experiment"] = "changed"

    assert result.metadata["experiment"] == "baseline"


def test_result_metadata_is_read_only() -> None:
    result = MonteCarloResult(
        method="estimate",
        metadata={"experiment": "baseline"},
    )

    assert isinstance(result.metadata, MappingProxyType)

    with pytest.raises(TypeError):
        result.metadata["new"] = "value"


def test_result_rejects_non_mapping_metadata() -> None:
    with pytest.raises(TypeError, match="mapping"):
        MonteCarloResult(
            method="estimate",
            metadata=["invalid"],
        )


# ============================================================================
# Dataclass immutability
# ============================================================================


def test_result_fields_cannot_be_reassigned() -> None:
    result = MonteCarloResult(method="estimate")

    with pytest.raises(FrozenInstanceError):
        result.method = "changed"


def test_result_has_slots() -> None:
    result = MonteCarloResult(method="estimate")

    assert not hasattr(result, "__dict__")


# ============================================================================
# Presence properties
# ============================================================================


def test_presence_properties_are_false_for_minimal_result() -> None:
    result = MonteCarloResult(method="empty")

    assert result.has_estimate is False
    assert result.has_samples is False
    assert result.has_path is False
    assert result.has_paths is False
    assert result.has_uncertainty is False


def test_has_estimate() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate=0.0,
    )

    assert result.has_estimate is True


def test_has_samples() -> None:
    result = MonteCarloResult(
        method="sample",
        samples=[],
        n_samples=0,
    )

    assert result.has_samples is True


def test_has_path() -> None:
    result = MonteCarloResult(
        method="simulate_chain",
        path=("a",),
        steps=0,
    )

    assert result.has_path is True


def test_has_paths() -> None:
    result = MonteCarloResult(
        method="simulate_paths",
        paths=(),
        n_paths=0,
    )

    assert result.has_paths is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance": 0.0},
        {"standard_error": 0.0},
        {
            "confidence_interval": (0.0, 1.0),
            "confidence_level": 0.95,
        },
    ],
)
def test_has_uncertainty_for_each_uncertainty_field(kwargs) -> None:
    result = MonteCarloResult(
        method="estimate",
        **kwargs,
    )

    assert result.has_uncertainty is True


# ============================================================================
# Derived properties
# ============================================================================


def test_interval_width_without_interval_is_none() -> None:
    result = MonteCarloResult(method="estimate")

    assert result.interval_width is None


def test_interval_width_is_upper_minus_lower() -> None:
    result = MonteCarloResult(
        method="estimate",
        confidence_interval=(0.25, 0.85),
        confidence_level=0.95,
    )

    assert result.interval_width == pytest.approx(0.60)


def test_effective_sample_size_prefers_explicit_n_samples() -> None:
    result = MonteCarloResult(
        method="estimate",
        n_samples=37,
        steps=100,
        burn_in=10,
        thinning=2,
    )

    assert result.effective_sample_size == 37


def test_effective_sample_size_is_none_without_counts() -> None:
    result = MonteCarloResult(method="estimate")

    assert result.effective_sample_size is None


def test_effective_sample_size_uses_all_observations_by_default() -> None:
    result = MonteCarloResult(
        method="estimate",
        steps=9,
    )

    assert result.effective_sample_size == 10


def test_effective_sample_size_applies_burn_in() -> None:
    result = MonteCarloResult(
        method="estimate",
        steps=9,
        burn_in=2,
    )

    assert result.effective_sample_size == 8


def test_effective_sample_size_applies_thinning_with_ceiling() -> None:
    result = MonteCarloResult(
        method="estimate",
        steps=9,
        burn_in=2,
        thinning=3,
    )

    assert result.effective_sample_size == 3


def test_effective_sample_size_with_burn_in_equal_to_steps() -> None:
    result = MonteCarloResult(
        method="estimate",
        steps=5,
        burn_in=5,
    )

    # The terminal observation remains after discarding five initial
    # observations from a six-observation path.
    assert result.effective_sample_size == 1


# ============================================================================
# Transformation methods
# ============================================================================


def test_with_metadata_returns_new_result() -> None:
    original = MonteCarloResult(
        method="estimate",
        estimate=0.5,
        metadata={"experiment": 1},
    )

    updated = original.with_metadata(
        experiment=2,
        note="updated",
    )

    assert updated is not original
    assert original.metadata["experiment"] == 1
    assert updated.metadata["experiment"] == 2
    assert updated.metadata["note"] == "updated"
    assert updated.estimate == pytest.approx(0.5)


def test_with_metadata_preserves_unmodified_metadata() -> None:
    original = MonteCarloResult(
        method="estimate",
        metadata={
            "a": 1,
            "b": 2,
        },
    )

    updated = original.with_metadata(c=3)

    assert dict(updated.metadata) == {
        "a": 1,
        "b": 2,
        "c": 3,
    }


def test_renamed_returns_new_result() -> None:
    original = MonteCarloResult(
        method="old_method",
        estimate=0.5,
    )

    renamed = original.renamed("new_method")

    assert renamed is not original
    assert renamed.method == "new_method"
    assert original.method == "old_method"
    assert renamed.estimate == pytest.approx(0.5)


def test_renamed_validates_new_method() -> None:
    result = MonteCarloResult(method="estimate")

    with pytest.raises(ValueError, match="nonempty"):
        result.renamed("   ")


# ============================================================================
# Summary
# ============================================================================


def test_summary_contains_expected_fields() -> None:
    result = MonteCarloResult(
        method="estimate_probability",
        estimate=0.5,
        samples=[0.0, 1.0],
        variance=0.25,
        standard_error=0.1,
        confidence_interval=(0.3, 0.7),
        confidence_level=0.95,
        n_samples=2,
        seed=42,
        rng_name="PCG64",
        converged=True,
        metadata={"target": "a"},
    )

    summary = result.summary()

    assert summary == {
        "method": "estimate_probability",
        "has_estimate": True,
        "has_samples": True,
        "has_path": False,
        "has_paths": False,
        "has_uncertainty": True,
        "n_samples": 2,
        "n_paths": None,
        "steps": None,
        "burn_in": None,
        "thinning": None,
        "effective_sample_size": 2,
        "seed": 42,
        "rng_name": "PCG64",
        "converged": True,
        "confidence_level": 0.95,
        "confidence_interval": (0.3, 0.7),
        "interval_width": pytest.approx(0.4),
        "metadata": {"target": "a"},
    }


def test_summary_returns_independent_metadata_dictionary() -> None:
    result = MonteCarloResult(
        method="estimate",
        metadata={"a": 1},
    )

    summary = result.summary()
    summary["metadata"]["a"] = 2

    assert result.metadata["a"] == 1


def test_summary_itself_is_mutable() -> None:
    result = MonteCarloResult(method="estimate")

    summary = result.summary()
    summary["method"] = "changed"

    assert result.method == "estimate"


# ============================================================================
# Equality
# ============================================================================


def test_identical_minimal_results_are_equal() -> None:
    left = MonteCarloResult(method="estimate")
    right = MonteCarloResult(method="estimate")

    assert left == right


def test_result_is_not_equal_to_unrelated_object() -> None:
    result = MonteCarloResult(method="estimate")

    assert result != object()


def test_results_with_equal_scalar_estimates_are_equal() -> None:
    left = MonteCarloResult(
        method="estimate",
        estimate=0.5,
    )
    right = MonteCarloResult(
        method="estimate",
        estimate=0.5,
    )

    assert left == right


def test_results_with_different_scalar_estimates_are_unequal() -> None:
    left = MonteCarloResult(
        method="estimate",
        estimate=0.5,
    )
    right = MonteCarloResult(
        method="estimate",
        estimate=0.6,
    )

    assert left != right


def test_results_with_equal_array_estimates_are_equal() -> None:
    left = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.25, 0.75]),
    )
    right = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.25, 0.75]),
    )

    assert left == right


def test_results_with_different_array_estimates_are_unequal() -> None:
    left = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.25, 0.75]),
    )
    right = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.30, 0.70]),
    )

    assert left != right


def test_results_with_array_nan_estimates_are_equal() -> None:
    left = MonteCarloResult(
        method="estimate",
        estimate=np.array([np.nan, 1.0]),
    )
    right = MonteCarloResult(
        method="estimate",
        estimate=np.array([np.nan, 1.0]),
    )

    assert left == right


def test_results_with_equal_samples_are_equal() -> None:
    left = MonteCarloResult(
        method="sample",
        samples=[1.0, 2.0],
        n_samples=2,
    )
    right = MonteCarloResult(
        method="sample",
        samples=[1.0, 2.0],
        n_samples=2,
    )

    assert left == right


def test_results_with_different_samples_are_unequal() -> None:
    left = MonteCarloResult(
        method="sample",
        samples=[1.0, 2.0],
        n_samples=2,
    )
    right = MonteCarloResult(
        method="sample",
        samples=[1.0, 3.0],
        n_samples=2,
    )

    assert left != right


def test_results_with_equal_variance_arrays_are_equal() -> None:
    left = MonteCarloResult(
        method="estimate",
        variance=[0.1, 0.2],
    )
    right = MonteCarloResult(
        method="estimate",
        variance=[0.1, 0.2],
    )

    assert left == right


def test_results_with_different_standard_errors_are_unequal() -> None:
    left = MonteCarloResult(
        method="estimate",
        standard_error=0.1,
    )
    right = MonteCarloResult(
        method="estimate",
        standard_error=0.2,
    )

    assert left != right


def test_results_with_different_metadata_are_unequal() -> None:
    left = MonteCarloResult(
        method="estimate",
        metadata={"version": 1},
    )
    right = MonteCarloResult(
        method="estimate",
        metadata={"version": 2},
    )

    assert left != right


@pytest.mark.parametrize(
    ("left_kwargs", "right_kwargs"),
    [
        (
            {"method": "a"},
            {"method": "b"},
        ),
        (
            {"method": "a", "states": ("x", "y")},
            {"method": "a", "states": ("y", "x")},
        ),
        (
            {"method": "a", "seed": 1},
            {"method": "a", "seed": 2},
        ),
        (
            {"method": "a", "converged": True},
            {"method": "a", "converged": False},
        ),
        (
            {"method": "a", "n_samples": 10},
            {"method": "a", "n_samples": 11},
        ),
    ],
)
def test_results_with_different_scalar_fields_are_unequal(
    left_kwargs,
    right_kwargs,
) -> None:
    assert (
        MonteCarloResult(**left_kwargs)
        != MonteCarloResult(**right_kwargs)
    )


# ============================================================================
# Representation
# ============================================================================


def test_minimal_repr_contains_class_and_method() -> None:
    result = MonteCarloResult(method="simulate_chain")

    representation = repr(result)

    assert representation.startswith("MonteCarloResult(")
    assert "method='simulate_chain'" in representation


def test_repr_contains_available_counts() -> None:
    result = MonteCarloResult(
        method="simulate_paths",
        paths=(
            ("a", "b"),
            ("b", "a"),
        ),
        steps=1,
        n_paths=2,
        seed=42,
    )

    representation = repr(result)

    assert "n_paths=2" in representation
    assert "steps=1" in representation
    assert "seed=42" in representation


def test_repr_marks_stored_estimate() -> None:
    result = MonteCarloResult(
        method="estimate",
        estimate=np.array([0.5, 0.5]),
    )

    assert "estimate=<stored>" in repr(result)


def test_repr_marks_stored_uncertainty() -> None:
    result = MonteCarloResult(
        method="estimate",
        variance=0.25,
    )

    assert "uncertainty=<stored>" in repr(result)


def test_repr_omits_absent_optional_fields() -> None:
    representation = repr(
        MonteCarloResult(method="estimate")
    )

    assert "n_samples=" not in representation
    assert "n_paths=" not in representation
    assert "steps=" not in representation
    assert "seed=" not in representation
    assert "estimate=<stored>" not in representation
    assert "uncertainty=<stored>" not in representation
