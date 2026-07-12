import numpy as np

import pytest

from spectral_operators.core.utilities import DimensionMismatchError
from spectral_operators.weights import (
    WeightOperator,
    PositionWeight,
    PolynomialWeight,
    ExponentialWeight,
    PrimeWeight,
    AdelicWeight,
)


def test_weight_operator_diagonal_matrix():
    W = WeightOperator([1, 2, 3])

    assert W.shape == (3, 3)
    assert np.allclose(W.matrix, np.diag([1, 2, 3]))
    assert np.allclose(W.weights, np.array([1, 2, 3]))


def test_position_weight_basic():
    W = PositionWeight(N=3, L=1.0, power=2, scale=1.0, offset=1.0)

    expected_grid = np.array([-1.0, 0.0, 1.0])
    expected_weights = np.array([2.0, 1.0, 2.0])

    assert np.allclose(W.grid, expected_grid)
    assert np.allclose(W.weights, expected_weights)
    assert np.allclose(W.matrix, np.diag(expected_weights))


def test_position_weight_normalized():
    W = PositionWeight(N=3, L=1.0, power=2, normalize=True)

    assert np.isclose(np.max(np.abs(W.weights)), 1.0)


def test_polynomial_weight_quadratic():
    W = PolynomialWeight(N=3, L=1.0, coefficients=[1, 0, 1])

    expected_weights = np.array([2.0, 1.0, 2.0])

    assert np.allclose(W.weights, expected_weights)


def test_polynomial_weight_normalized():
    W = PolynomialWeight(
        N=3,
        L=1.0,
        coefficients=[1, 0, 1],
        normalize=True,
    )

    assert np.isclose(np.max(np.abs(W.weights)), 1.0)


def test_exponential_weight_basic():
    W = ExponentialWeight(
        N=3,
        L=1.0,
        rate=-1.0,
        power=2,
        scale=1.0,
        offset=0.0,
    )

    expected_weights = np.exp(-np.array([1.0, 0.0, 1.0]))

    assert np.allclose(W.weights, expected_weights)


def test_exponential_weight_normalized():
    W = ExponentialWeight(N=3, L=1.0, normalize=True)

    assert np.isclose(np.max(np.abs(W.weights)), 1.0)


def test_prime_weight_inverse_normalized():
    P = PrimeWeight([2, 3], rule="inverse", normalize=True)

    raw = np.array([1 / 2, 1 / 3])
    expected = raw / np.sum(np.abs(raw))

    assert np.allclose(P.weights, expected)


def test_prime_weight_as_dict():
    P = PrimeWeight([2, 3], rule="uniform", normalize=False)

    assert P.as_dict() == {2: 1.0, 3: 1.0}


def test_adelic_weight_basic():
    A = AdelicWeight(labels=["a", "b"], weights=[1, 3], normalize=True)

    assert A.labels == ("a", "b")
    assert np.allclose(A.weights, np.array([0.25, 0.75]))


def test_adelic_weight_from_primes():
    A = AdelicWeight.from_primes([2, 3], rule="uniform", normalize=True)

    assert A.labels == (2, 3)
    assert np.allclose(A.weights, np.array([0.5, 0.5]))


def test_adelic_weight_as_array_returns_copy():
    A = AdelicWeight(labels=["a", "b"], weights=[1, 1])
    arr = A.as_array()
    arr[0] = 99

    assert not np.allclose(arr, A.weights)

def test_weight_operator_weights_are_read_only():
    weight = WeightOperator([1, 2, 3])

    with pytest.raises(ValueError):
        weight.weights[0] = 10


def test_position_weight_grid_is_read_only():
    weight = PositionWeight(N=3, L=1.0)

    with pytest.raises(ValueError):
        weight.grid[0] = 10


def test_adelic_weight_requires_matching_length():
    with pytest.raises(DimensionMismatchError):
        AdelicWeight(
            labels=["a", "b"],
            weights=[1.0],
        )
