from spectral_operators.constants import (
    AUTHOR,
    DEFAULT_BOUNDARY_WIDTH,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    DEFAULT_NORMALIZE,
    DEFAULT_NORM,
    DEFAULT_ORDERING,
    DEFAULT_TOL,
    DEFAULT_ZERO_TOL,
    EvolutionType,
    LICENSE,
    Ordering,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PROJECT_NAME,
    WeightRule,
)


def test_default_tolerances_are_positive():
    assert DEFAULT_TOL > 0
    assert DEFAULT_ZERO_TOL > 0


def test_default_names():
    assert DEFAULT_NORM == "fro"
    assert DEFAULT_ORDERING == "abs"
    assert DEFAULT_BOUNDARY_WIDTH == 1
    assert DEFAULT_NORMALIZE is True


def test_plotting_defaults():
    assert DEFAULT_FIGSIZE == (8, 6)
    assert DEFAULT_DPI == 150


def test_package_metadata():
    assert PROJECT_NAME == "operatorlab"
    assert PACKAGE_NAME == "spectral_operators"
    assert isinstance(PACKAGE_VERSION, str)
    assert AUTHOR == "Salma Y. Rodriguez"
    assert LICENSE == "MIT"


def test_ordering_enum_values():
    assert Ordering.ABS.value == "abs"
    assert Ordering.REAL.value == "real"
    assert Ordering.IMAG.value == "imag"
    assert Ordering.COMPLEX.value == "complex"


def test_weight_rule_enum_values():
    assert WeightRule.UNIFORM.value == "uniform"
    assert WeightRule.INVERSE.value == "inverse"
    assert WeightRule.LOG.value == "log"
    assert WeightRule.LOG_INVERSE.value == "log_inverse"


def test_evolution_type_enum_values():
    assert EvolutionType.UNITARY.value == "unitary"
    assert EvolutionType.SEMIGROUP.value == "semigroup"

def test_default_ordering_is_supported():
    assert DEFAULT_ORDERING in {
        ordering.value
        for ordering in Ordering
    }


def test_default_norm_is_supported():
    assert DEFAULT_NORM == "fro"
