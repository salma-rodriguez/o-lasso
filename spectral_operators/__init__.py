"""
Spectral operator constructions and analysis tools.

This package provides the public interface for OperatorLab's
spectral-operator functionality.
"""

from .adelic import (
    AdelicAnalyzer,
    AdelicBuilder,
    AdelicSystem,
    LocalComponent,
)

from .constants import (
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

from .core import (
    DimensionMismatchError,
    Field,
    LinearOperator,
    NonSquareOperatorError,
    Norm,
    OperatorBase,
    OperatorError,
    OperatorFactory,
    SingularOperatorError,
)

from .diagnostics import (
    ComparativeDiagnostics,
    DiagnosticReport,
    OperatorDiagnostics,
    StabilityDiagnostics,
)

from .evolution import (
    EvolutionAnalyzer,
    Propagator,
    SemigroupEvolution,
    UnitaryEvolution,
)

from .geometry import (
    BoundaryAnalyzer,
    GeometryAnalyzer,
    LocalityAnalyzer,
    SymmetryAnalyzer,
)

from .io import (
    DiagnosticIO,
    JSONSerializer,
    NumpyIO,
    OperatorIO,
)

from .operators import (
    AdelicOperator,
    FiniteDifferenceOperator,
    GradedOperator,
    HamiltonianOperator,
    WeightedOperator,
    ZetaOperator,
)

from .spectrum import (
    ResolventAnalyzer,
    SpectralAnalyzer,
    SpectralPartition,
    SpectralStatistics,
)

from .visualization import (
    GeometryData,
    MatrixData,
    SpectrumData,
    VisualizationBundle,
    ZetaData,
)

from .weights import (
    AdelicWeight,
    ExponentialWeight,
    PolynomialWeight,
    PositionWeight,
    PrimeWeight,
    WeightOperator,
)

from .zeta import (
    HilbertPolyaAnalyzer,
    SpectralZeta,
    ZetaCorrespondence,
    ZetaZeroSet,
)

__all__ = [
    # adelic
    "AdelicAnalyzer",
    "AdelicBuilder",
    "AdelicSystem",
    "LocalComponent",

    # constants
    "AUTHOR",
    "DEFAULT_BOUNDARY_WIDTH",
    "DEFAULT_DPI",
    "DEFAULT_FIGSIZE",
    "DEFAULT_NORMALIZE",
    "DEFAULT_NORM",
    "DEFAULT_ORDERING",
    "DEFAULT_TOL",
    "DEFAULT_ZERO_TOL",
    "EvolutionType",
    "LICENSE",
    "Ordering",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "PROJECT_NAME",
    "WeightRule",

    # core
    "DimensionMismatchError",
    "Field",
    "LinearOperator",
    "NonSquareOperatorError",
    "Norm",
    "OperatorBase",
    "OperatorError",
    "OperatorFactory",
    "SingularOperatorError",

    # diagnostics
    "ComparativeDiagnostics",
    "DiagnosticReport",
    "OperatorDiagnostics",
    "StabilityDiagnostics",

    # evolution
    "EvolutionAnalyzer",
    "Propagator",
    "SemigroupEvolution",
    "UnitaryEvolution",

    # geometry
    "BoundaryAnalyzer",
    "GeometryAnalyzer",
    "LocalityAnalyzer",
    "SymmetryAnalyzer",

    # io
    "DiagnosticIO",
    "JSONSerializer",
    "NumpyIO",
    "OperatorIO",

    # operators
    "AdelicOperator",
    "FiniteDifferenceOperator",
    "GradedOperator",
    "HamiltonianOperator",
    "WeightedOperator",
    "ZetaOperator",

    # visualization
    "GeometryData",
    "MatrixData",
    "SpectrumData",
    "VisualizationBundle",
    "ZetaData",

    # weights
    "AdelicWeight",
    "ExponentialWeight",
    "PolynomialWeight",
    "PositionWeight",
    "PrimeWeight",
    "WeightOperator",

    # zeta
    "HilbertPolyaAnalyzer",
    "SpectralZeta",
    "ZetaCorrespondence",
    "ZetaZeroSet",


]
