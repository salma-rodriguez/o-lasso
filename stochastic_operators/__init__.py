"""
Stochastic operator constructions and analysis tools for OperatorLab.
"""

from .ergodic import ErgodicAnalyzer
from .generators import MarkovGenerator
from .hitting import HittingAnalyzer
from .kernels import Kernel, StochasticKernel
from .markov import MarkovOperator
from .monte_carlo import (
    MonteCarloResult,
    empirical_distribution,
    empirical_hitting_probability,
    empirical_hitting_time,
    empirical_return_time,
    empirical_stationary_distribution,
    empirical_transition_matrix,
    simulate_chain,
    simulate_paths,
)
from .operators import (
    StochasticConvention,
    StochasticOperator,
)
from .stationary import StationaryAnalyzer

__all__ = [
    "ErgodicAnalyzer",
    "HittingAnalyzer",
    "Kernel",
    "MarkovGenerator",
    "MarkovOperator",
    "MonteCarloResult",
    "empirical_distribution",
    "empirical_hitting_probability",
    "empirical_hitting_time",
    "empirical_return_time",
    "empirical_stationary_distribution",
    "empirical_transition_matrix",
    "simulate_chain",
    "simulate_paths",
    "StochasticAnalyzer",
    "StochasticConvention",
    "StochasticKernel",
    "StochasticOperator",
]
