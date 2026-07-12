import numpy as np
import pytest

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.zeta import ZetaZeroSet, ZetaCorrespondence
from spectral_operators import (
    GeometryData,
    LinearOperator,
    MatrixData,
    SpectrumData,
    VisualizationBundle,
    ZetaCorrespondence,
    ZetaData,
    ZetaZeroSet,
)


def test_spectrum_data_complex_points():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    S = SpectrumData(A, ordering="real")

    expected = np.array([[1.0, 0.0], [2.0, 0.0]])

    assert np.allclose(S.complex_points(), expected)


def test_spectrum_data_as_dict():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    S = SpectrumData(A, ordering="real")

    d = S.as_dict()

    assert d["operator"] == "A"
    assert d["ordering"] == "real"
    assert d["real"] == [1.0, 2.0]


def test_matrix_data_components():
    A = LinearOperator(np.array([[1 + 1j, 2], [3, 4 - 1j]]), name="A")
    M = MatrixData(A)

    assert np.allclose(M.real(), np.array([[1, 2], [3, 4]]))
    assert np.allclose(M.imag(), np.array([[1, 0], [0, -1]]))


def test_matrix_data_as_dict():
    A = LinearOperator(np.eye(2), name="I")
    M = MatrixData(A)

    d = M.as_dict()

    assert d["operator"] == "I"
    assert d["shape"] == (2, 2)
    assert d["real"] == [[1.0, 0.0], [0.0, 1.0]]


def test_geometry_data_as_dict():
    A = LinearOperator(np.eye(2), name="I")
    G = GeometryData(A)

    d = G.as_dict()

    assert d["operator"] == "I"
    assert "defects" in d
    assert "ratios" in d


def test_zeta_data_paired_points_and_errors():
    A = LinearOperator(np.diag([14.0, 21.0]), name="A")
    zeros = ZetaZeroSet([14.0, 21.0])
    C = ZetaCorrespondence(A, zeros, ordering="real")

    Z = ZetaData(C)

    assert np.allclose(Z.paired_points(), np.array([[14.0, 14.0], [21.0, 21.0]]))
    assert np.allclose(Z.error_series(), np.array([0.0, 0.0]))


def test_zeta_data_as_dict():
    A = LinearOperator(np.diag([14.0, 21.0]), name="A")
    zeros = ZetaZeroSet([14.0, 21.0])
    C = ZetaCorrespondence(A, zeros, ordering="real")

    Z = ZetaData(C)
    d = Z.as_dict()

    assert d["count"] == 2
    assert d["spectral_values"] == [14.0, 21.0]
    assert d["zero_ordinates"] == [14.0, 21.0]


def test_visualization_bundle_components():
    A = LinearOperator(np.eye(2), name="I")
    V = VisualizationBundle(A)

    assert isinstance(V.spectrum(), SpectrumData)
    assert isinstance(V.matrix(), MatrixData)
    assert isinstance(V.geometry(), GeometryData)


def test_visualization_bundle_as_dict():
    A = LinearOperator(np.eye(2), name="I")
    V = VisualizationBundle(A)

    d = V.as_dict(ordering="real")

    assert d["operator"] == "I"
    assert "spectrum" in d
    assert "matrix" in d
    assert "geometry" in d
    assert "diagnostics" in d


def test_spectrum_points_are_read_only():
    operator = LinearOperator(
        np.diag([1.0, 2.0])
    )
    data = SpectrumData(
        operator,
        ordering="real",
    )

    points = data.complex_points()

    with pytest.raises(ValueError):
        points[0, 0] = 99.0


def test_matrix_components_are_read_only():
    operator = LinearOperator(
        np.eye(2)
    )
    data = MatrixData(operator)

    real = data.real()

    with pytest.raises(ValueError):
        real[0, 0] = 99.0


def test_visualization_bundle_preserves_boundary_width():
    operator = LinearOperator(
        np.eye(4)
    )
    bundle = VisualizationBundle(operator)

    data = bundle.as_dict(
        boundary_width=2,
        bandwidth=0,
    )

    assert (
        data["geometry"]["boundary_width"]
        == 2
    )


def test_zeta_error_series_is_read_only():
    operator = LinearOperator(
        np.diag([14.0, 21.0])
    )
    zeros = ZetaZeroSet(
        [14.0, 21.0]
    )
    correspondence = ZetaCorrespondence(
        operator,
        zeros,
        ordering="real",
    )

    errors = ZetaData(
        correspondence
    ).error_series()

    with pytest.raises(ValueError):
        errors[0] = 99.0
