"""
Monte Carlo simulation and empirical estimation.

The subpackage is organized into four layers:

- result: immutable result objects;
- simulation: trajectory generation;
- empirical: empirical estimators;
- statistics: statistical summaries and confidence intervals.
"""

from .empirical import (
    empirical_distribution,
    empirical_hitting_probability,
    empirical_hitting_time,
    empirical_return_time,
    empirical_stationary_distribution,
    empirical_transition_matrix,
)

from .result import (
    ConfidenceInterval,
    MonteCarloResult,
    Path,
    Paths,
    State,
)
from .simulation import (
    simulate_chain,
    simulate_paths,
)


__all__ = [
    "ConfidenceInterval",
    "MonteCarloResult",
    "Path",
    "Paths",
    "State",
    "simulate_chain",
    "simulate_paths",
    "empirical_distribution",
    "empirical_transition_matrix",
    "empirical_stationary_distribution",
    "empirical_hitting_probability",
    "empirical_hitting_time",
    "empirical_return_time",
]
