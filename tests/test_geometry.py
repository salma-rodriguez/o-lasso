import numpy as np
import pytest

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.core.exceptions import OperatorError
from spectral_operators.core.utilities import (
    readonly_array,
    require_nonnegative_integer,
    require_positive_integer,
    require_probability
)
from spectral_operators.geometry import (
    SymmetryAnalyzer,
    BoundaryAnalyzer,
    LocalityAnalyzer,
    GeometryAnalyzer,
)


def test_symmetry_analyzer_detects_symmetric_matrix():
    A = LinearOperator(np.array([[1, 2], [2, 3]], dtype=float))
    G = SymmetryAnalyzer(A)

    assert G.symmetric_defect() == 0.0
    assert G.hermitian_defect() == 0.0
    assert G.summary()["symmetric"] is True


def test_symmetry_analyzer_detects_skew_matrix():
    A = LinearOperator(np.array([[0, 2], [-2, 0]], dtype=float))
    G = SymmetryAnalyzer(A)

    assert G.skew_defect() == 0.0
    assert G.antihermitian_defect() == 0.0
    assert G.summary()["skew"] is True


def test_symmetry_relative_defect_zero_operator():
    A = LinearOperator(np.zeros((2, 2)))
    G = SymmetryAnalyzer(A)

    assert G.symmetric_defect(relative=True) == 0.0
    assert G.hermitian_defect(relative=True) == 0.0


def test_boundary_mask_width_one():
    A = LinearOperator(np.ones((4, 4)))
    B = BoundaryAnalyzer(A, width=1)

    mask = B.boundary_mask()

    assert mask.shape == (4, 4)
    assert np.sum(mask) == 12


def test_boundary_and_interior_ratios():
    A = LinearOperator(np.ones((4, 4)))
    B = BoundaryAnalyzer(A, width=1)

    assert np.isclose(B.boundary_ratio(), np.sqrt(12) / 4)
    assert np.isclose(B.interior_ratio(), 2 / 4)


def test_locality_band_mask_diagonal():
    A = LinearOperator(np.eye(4))
    L = LocalityAnalyzer(A)

    mask = L.band_mask(0)

    assert np.sum(mask) == 4
    assert L.locality_ratio(0) == 1.0


def test_locality_off_band_norm_zero_for_identity():
    A = LinearOperator(np.eye(4))
    L = LocalityAnalyzer(A)

    assert L.off_band_norm(0) == 0.0
    assert L.off_locality_ratio(0) == 0.0


def test_effective_bandwidth_identity():
    A = LinearOperator(np.eye(5))
    L = LocalityAnalyzer(A)

    assert L.effective_bandwidth(0.95) == 0


def test_geometry_analyzer_summary():
    A = LinearOperator(np.eye(4), name="I")
    G = GeometryAnalyzer(A, boundary_width=1)

    summary = G.summary(bandwidth=0)

    assert summary["operator"] == "I"
    assert summary["shape"] == (4, 4)
    assert "symmetry" in summary
    assert "boundary" in summary
    assert "locality" in summary


def test_geometry_analyzer_ratios():
    A = LinearOperator(np.eye(4))
    G = GeometryAnalyzer(A, boundary_width=1)

    ratios = G.ratios(bandwidth=0)

    assert ratios["relative_symmetric_defect"] == 0.0
    assert ratios["relative_hermitian_defect"] == 0.0
    assert ratios["locality_ratio"] == 1.0


def test_boundary_mask_is_read_only():
    operator = LinearOperator(
        np.ones((4, 4))
    )
    analyzer = BoundaryAnalyzer(
        operator,
        width=1,
    )

    mask = analyzer.boundary_mask()

    with pytest.raises(ValueError):
        mask[0, 0] = False


def test_negative_bandwidth_rejected():
    operator = LinearOperator(
        np.eye(3)
    )
    analyzer = LocalityAnalyzer(operator)

    with pytest.raises(OperatorError):
        analyzer.band_mask(-1)


def test_effective_bandwidth_dense_matrix():
    operator = LinearOperator(
        np.ones((3, 3))
    )
    analyzer = LocalityAnalyzer(operator)

    assert analyzer.effective_bandwidth(
        threshold=1.0
    ) == 2
