import numpy as np
import pytest

from spectral_operators.core.algebra import LinearOperator
from spectral_operators.zeta import (
    SpectralZeta,
    ZetaZeroSet,
    ZetaCorrespondence,
    HilbertPolyaAnalyzer,
)


def test_spectral_zeta_evaluate():
    A = LinearOperator(np.diag([1.0, 2.0, 4.0]), name="A")
    Z = SpectralZeta(A)

    expected = 1.0 + 1 / 2.0 + 1 / 4.0

    assert np.isclose(Z.evaluate(1), expected)


def test_spectral_zeta_discards_zero_eigenvalues():
    A = LinearOperator(np.diag([0.0, 1.0, 2.0]), name="A")
    Z = SpectralZeta(A)

    assert len(Z.eigenvalues) == 2


def test_spectral_zeta_values():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    Z = SpectralZeta(A)

    vals = Z.values([1, 2])

    assert np.allclose(vals, np.array([1.5, 1.25]))


def test_zeta_zero_set_zeros():
    zeros = ZetaZeroSet([14.0, 21.0])

    expected = np.array([0.5 + 14.0j, 0.5 + 21.0j])

    assert np.allclose(zeros.zeros, expected)


def test_zeta_zero_set_first_and_spacings():
    zeros = ZetaZeroSet([21.0, 14.0, 25.0])

    assert np.allclose(zeros.first(2), np.array([14.0, 21.0]))
    assert np.allclose(zeros.spacings(), np.array([7.0, 4.0]))


def test_zeta_zero_set_summary():
    zeros = ZetaZeroSet([14.0, 21.0, 25.0])
    summary = zeros.summary()

    assert summary["count"] == 3
    assert summary["min_gamma"] == 14.0
    assert summary["max_gamma"] == 25.0
    assert np.isclose(summary["mean_spacing"], 5.5)


def test_zeta_correspondence_compare_real_spectrum():
    A = LinearOperator(np.diag([14.0, 21.0, 25.0]), name="A")
    zeros = ZetaZeroSet([14.0, 21.0, 25.0])

    C = ZetaCorrespondence(A, zeros, ordering="real")
    result = C.compare()

    assert result["count"] == 3
    assert np.isclose(result["mean_abs_error"], 0.0)
    assert np.isclose(result["rms_error"], 0.0)


def test_zeta_correspondence_paired_values():
    A = LinearOperator(np.diag([14.0, 21.0, 25.0]), name="A")
    zeros = ZetaZeroSet([14.0, 21.0, 25.0])

    C = ZetaCorrespondence(A, zeros, ordering="real")
    spec, gamma = C.paired_values(n=2)

    assert np.allclose(spec, np.array([14.0, 21.0]))
    assert np.allclose(gamma, np.array([14.0, 21.0]))


def test_hilbert_polya_analyzer_detects_hermitian():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    H = HilbertPolyaAnalyzer(A)

    assert H.is_candidate_self_adjoint()
    assert np.isclose(H.real_spectrum_defect(), 0.0)


def test_hilbert_polya_analyzer_summary():
    A = LinearOperator(np.diag([1.0, 2.0]), name="A")
    H = HilbertPolyaAnalyzer(A)

    summary = H.summary()

    assert summary["operator"] == "A"
    assert summary["shape"] == (2, 2)
    assert summary["is_hermitian"] is True
    assert summary["num_eigenvalues"] == 2

from spectral_operators import (
    HilbertPolyaAnalyzer,
    LinearOperator,
    SpectralZeta,
    ZetaCorrespondence,
    ZetaZeroSet,
)
from spectral_operators.core.exceptions import OperatorError


def test_zeta_zero_ordinates_are_read_only():
    zeros = ZetaZeroSet([14.0, 21.0])

    with pytest.raises(ValueError):
        zeros.gammas[0] = 99.0


def test_zeta_zero_set_rejects_negative_ordinates():
    with pytest.raises(OperatorError):
        ZetaZeroSet([-14.0])


def test_paired_values_rejects_negative_count():
    operator = LinearOperator(
        np.diag([14.0, 21.0])
    )
    zeros = ZetaZeroSet([14.0, 21.0])
    correspondence = ZetaCorrespondence(
        operator,
        zeros,
        ordering="real",
    )

    with pytest.raises(OperatorError):
        correspondence.paired_values(-1)


def test_spectral_zeta_rejects_nonpositive_tolerance():
    operator = LinearOperator(np.eye(2))

    with pytest.raises(OperatorError):
        SpectralZeta(
            operator,
            zero_tol=0.0,
        )
