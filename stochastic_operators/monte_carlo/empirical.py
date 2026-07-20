"""
Empirical estimators for finite-state stochastic processes.

This module consumes observed or simulated state paths and constructs
empirical estimates such as

- state distributions;
- transition matrices;
- stationary distributions;
- hitting probabilities;
- hitting times;
- return times.

The routines in this module do not generate stochastic paths. Path
generation belongs to ``simulation.py``. Confidence intervals and more
general statistical summaries belong to ``statistics.py``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from numbers import Integral
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .result import MonteCarloResult


State: TypeAlias = Hashable
Path: TypeAlias = tuple[State, ...]
Paths: TypeAlias = tuple[Path, ...]
FloatArray: TypeAlias = NDArray[np.float64]
IntegerArray: TypeAlias = NDArray[np.int64]

ZeroRowPolicy: TypeAlias = Literal[
    "nan",
    "zeros",
    "self",
]


# ============================================================================
# Basic validation
# ============================================================================


def _validate_nonnegative_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a nonnegative integer.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Integral, np.integer),
    ):
        raise TypeError(
            f"{name} must be a nonnegative integer."
        )

    result = int(value)

    if result < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )

    return result


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> int:
    """
    Validate a positive integer.
    """

    result = _validate_nonnegative_integer(
        value,
        name=name,
    )

    if result == 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _validate_boolean(
    value: bool,
    *,
    name: str,
) -> bool:
    """
    Validate a Boolean value.
    """

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a Boolean value."
        )

    return bool(value)


def _validate_zero_row_policy(
    policy: ZeroRowPolicy,
) -> ZeroRowPolicy:
    """
    Validate the handling policy for unobserved transition rows.
    """

    valid = {
        "nan",
        "zeros",
        "self",
    }

    if not isinstance(policy, str):
        raise TypeError(
            "zero_row must be a string."
        )

    if policy not in valid:
        raise ValueError(
            "zero_row must be one of "
            "{'nan', 'zeros', 'self'}."
        )

    return policy


# ============================================================================
# State and path validation
# ============================================================================


def _freeze_state_path(
    path: Sequence[State],
    *,
    name: str = "path",
    allow_empty: bool = False,
) -> Path:
    """
    Validate and freeze a finite state path.
    """

    if isinstance(path, (str, bytes)):
        raise TypeError(
            f"{name} must be a sequence of states, not a string."
        )

    try:
        frozen = tuple(path)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a finite sequence of states."
        ) from exc

    if not frozen and not allow_empty:
        raise ValueError(
            f"{name} must be nonempty."
        )

    for state in frozen:
        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                f"Every state in {name} must be hashable."
            ) from exc

    return frozen


def _freeze_states(
    states: Sequence[State],
) -> tuple[State, ...]:
    """
    Validate and freeze an ordered state space.
    """

    frozen = _freeze_state_path(
        states,
        name="states",
        allow_empty=False,
    )

    if len(set(frozen)) != len(frozen):
        raise ValueError(
            "states must contain unique labels."
        )

    return frozen


def _extract_paths(
    data: MonteCarloResult | Sequence[State],
) -> Paths:
    """
    Extract paths from a simulation result or a single raw path.

    Raw input is interpreted as one path. Multiple raw paths are deliberately
    not inferred because tuple-valued states would make such inference
    ambiguous. Multiple paths should therefore be supplied through a
    ``MonteCarloResult`` returned by ``simulate_paths``.
    """

    if isinstance(data, MonteCarloResult):
        if data.path is not None:
            return (
                _freeze_state_path(
                    data.path,
                    name="result.path",
                ),
            )

        if data.paths is not None:
            if len(data.paths) == 0:
                raise ValueError(
                    "MonteCarloResult.paths must contain at least one path."
                )

            return tuple(
                _freeze_state_path(
                    path,
                    name=f"result.paths[{index}]",
                )
                for index, path in enumerate(data.paths)
            )

        raise ValueError(
            "MonteCarloResult contains neither path nor paths."
        )

    return (
        _freeze_state_path(
            data,
            name="path",
        ),
    )


def _infer_states(
    paths: Paths,
) -> tuple[State, ...]:
    """
    Infer state labels in first-observation order.
    """

    ordered: list[State] = []
    observed: set[State] = set()

    for path in paths:
        for state in path:
            if state not in observed:
                observed.add(state)
                ordered.append(state)

    if not ordered:
        raise ValueError(
            "Cannot infer states from empty paths."
        )

    return tuple(ordered)


def _resolve_states(
    paths: Paths,
    *,
    states: Sequence[State] | None,
) -> tuple[State, ...]:
    """
    Resolve the state ordering and validate all observations.
    """

    if states is None:
        return _infer_states(paths)

    frozen_states = _freeze_states(states)
    state_set = set(frozen_states)

    for path_index, path in enumerate(paths):
        for observation_index, state in enumerate(path):
            if state not in state_set:
                raise ValueError(
                    "Observed state "
                    f"{state!r} at path {path_index}, "
                    f"position {observation_index} is not present "
                    "in states."
                )

    return frozen_states


def _trim_paths(
    paths: Paths,
    *,
    burn_in: int,
    thinning: int,
    minimum_length: int = 1,
) -> Paths:
    """
    Apply burn-in and thinning independently to each path.
    """

    burn_in = _validate_nonnegative_integer(
        burn_in,
        name="burn_in",
    )
    thinning = _validate_positive_integer(
        thinning,
        name="thinning",
    )
    minimum_length = _validate_positive_integer(
        minimum_length,
        name="minimum_length",
    )

    trimmed: list[Path] = []

    for index, path in enumerate(paths):
        processed = path[burn_in::thinning]

        if len(processed) < minimum_length:
            raise ValueError(
                f"Path {index} contains fewer than "
                f"{minimum_length} retained observations after "
                "burn-in and thinning."
            )

        trimmed.append(processed)

    return tuple(trimmed)


def _flatten_paths(
    paths: Paths,
) -> Path:
    """
    Flatten independent paths without introducing artificial transitions.
    """

    return tuple(
        state
        for path in paths
        for state in path
    )


def _state_index(
    states: tuple[State, ...],
) -> dict[State, int]:
    """
    Construct a state-to-index mapping.
    """

    return {
        state: index
        for index, state in enumerate(states)
    }


# ============================================================================
# Numerical helpers
# ============================================================================


def _readonly_float_array(
    values: Any,
) -> FloatArray:
    """
    Create an independent read-only float array.
    """

    result = np.array(
        values,
        dtype=np.float64,
        copy=True,
    )
    result.setflags(write=False)

    return result


def _readonly_integer_array(
    values: Any,
) -> IntegerArray:
    """
    Create an independent read-only integer array.
    """

    result = np.array(
        values,
        dtype=np.int64,
        copy=True,
    )
    result.setflags(write=False)

    return result


def _sample_variance(
    samples: FloatArray,
) -> float | None:
    """
    Return the unbiased sample variance when at least two samples exist.
    """

    if samples.size < 2:
        return None

    return float(
        np.var(
            samples,
            ddof=1,
        )
    )


def _standard_error(
    variance: float | None,
    *,
    n_samples: int,
) -> float | None:
    """
    Compute the standard error of a sample mean.
    """

    if variance is None or n_samples == 0:
        return None

    return float(
        np.sqrt(
            variance / n_samples
        )
    )


def _mean_or_nan(
    samples: FloatArray,
) -> float:
    """
    Return a sample mean or NaN when no samples are available.
    """

    if samples.size == 0:
        return float("nan")

    return float(np.mean(samples))


# ============================================================================
# Empirical state distribution
# ============================================================================


def empirical_distribution(
    data: MonteCarloResult | Sequence[State],
    *,
    states: Sequence[State] | None = None,
    burn_in: int = 0,
    thinning: int = 1,
) -> MonteCarloResult:
    """
    Estimate a finite-state empirical distribution.

    Every retained observation contributes one count. When several paths are
    supplied through a ``MonteCarloResult``, observations are pooled across
    paths.

    Parameters
    ----------
    data
        A simulation result or one raw state path.

    states
        Optional explicit state ordering. When omitted, states are inferred
        in first-observation order.

    burn_in
        Number of initial observations removed from every path.

    thinning
        Retain every ``thinning``-th observation after burn-in.

    Returns
    -------
    MonteCarloResult
        Result whose estimate is a probability vector ordered according to
        ``result.states``.
    """

    paths = _extract_paths(data)

    burn_in = _validate_nonnegative_integer(
        burn_in,
        name="burn_in",
    )
    thinning = _validate_positive_integer(
        thinning,
        name="thinning",
    )

    frozen_states = _resolve_states(
        paths,
        states=states,
    )

    retained_paths = _trim_paths(
        paths,
        burn_in=burn_in,
        thinning=thinning,
    )

    observations = _flatten_paths(retained_paths)
    index = _state_index(frozen_states)

    counts = np.zeros(
        len(frozen_states),
        dtype=np.int64,
    )

    for state in observations:
        counts[index[state]] += 1

    probabilities = counts.astype(np.float64)
    probabilities /= probabilities.sum()

    estimate = _readonly_float_array(probabilities)

    return MonteCarloResult(
        method="empirical_distribution",
        estimate=estimate,
        states=frozen_states,
        n_samples=len(observations),
        n_paths=len(retained_paths),
        burn_in=burn_in,
        thinning=thinning,
        metadata={
            "counts": tuple(
                int(value)
                for value in counts
            ),
            "pooled_paths": len(retained_paths) > 1,
            "estimator": "relative_frequency",
        },
    )


# ============================================================================
# Empirical transition matrix
# ============================================================================


def empirical_transition_matrix(
    data: MonteCarloResult | Sequence[State],
    *,
    states: Sequence[State] | None = None,
    burn_in: int = 0,
    thinning: int = 1,
    zero_row: ZeroRowPolicy = "nan",
) -> MonteCarloResult:
    """
    Estimate a row-stochastic transition matrix from observed paths.

    Transitions are counted only within individual paths. No artificial
    transition is introduced between the end of one path and the beginning
    of another.

    Parameters
    ----------
    data
        A simulation result or one raw state path.

    states
        Optional state ordering.

    burn_in
        Number of initial observations removed from every path.

    thinning
        Retain every ``thinning``-th observation after burn-in.

    zero_row
        Policy for states with no observed outgoing transitions:

        ``"nan"``
            Fill the row with NaN values.

        ``"zeros"``
            Fill the row with zeros.

        ``"self"``
            Insert a deterministic self-transition.

    Returns
    -------
    MonteCarloResult
        Result whose estimate is an empirical transition matrix.
    """

    paths = _extract_paths(data)

    burn_in = _validate_nonnegative_integer(
        burn_in,
        name="burn_in",
    )
    thinning = _validate_positive_integer(
        thinning,
        name="thinning",
    )
    zero_row = _validate_zero_row_policy(zero_row)

    frozen_states = _resolve_states(
        paths,
        states=states,
    )

    retained_paths = _trim_paths(
        paths,
        burn_in=burn_in,
        thinning=thinning,
        minimum_length=1,
    )

    index = _state_index(frozen_states)
    n_states = len(frozen_states)

    counts = np.zeros(
        (n_states, n_states),
        dtype=np.int64,
    )

    for path in retained_paths:
        for current_state, next_state in zip(
            path[:-1],
            path[1:],
        ):
            counts[
                index[current_state],
                index[next_state],
            ] += 1

    row_totals = counts.sum(axis=1)
    matrix = np.empty(
        (n_states, n_states),
        dtype=np.float64,
    )

    unobserved_rows: list[State] = []

    for row in range(n_states):
        total = int(row_totals[row])

        if total > 0:
            matrix[row] = counts[row] / total
            continue

        unobserved_rows.append(frozen_states[row])

        if zero_row == "nan":
            matrix[row] = np.nan
        elif zero_row == "zeros":
            matrix[row] = 0.0
        else:
            matrix[row] = 0.0
            matrix[row, row] = 1.0

    transition_count = int(counts.sum())

    return MonteCarloResult(
        method="empirical_transition_matrix",
        estimate=_readonly_float_array(matrix),
        states=frozen_states,
        n_samples=transition_count,
        n_paths=len(retained_paths),
        burn_in=burn_in,
        thinning=thinning,
        metadata={
            "transition_counts": tuple(
                tuple(
                    int(value)
                    for value in row
                )
                for row in counts
            ),
            "row_totals": tuple(
                int(value)
                for value in row_totals
            ),
            "zero_row": zero_row,
            "unobserved_rows": tuple(unobserved_rows),
            "pooled_paths": len(retained_paths) > 1,
        },
    )


# ============================================================================
# Empirical stationary distribution
# ============================================================================


def empirical_stationary_distribution(
    data: MonteCarloResult | Sequence[State],
    *,
    states: Sequence[State] | None = None,
    burn_in: int = 0,
    thinning: int = 1,
) -> MonteCarloResult:
    """
    Estimate a stationary distribution using occupation frequencies.

    This routine does not assert that the observed process is stationary,
    irreducible, or ergodic. It merely computes the empirical occupation
    measure after optional burn-in and thinning.
    """

    distribution = empirical_distribution(
        data,
        states=states,
        burn_in=burn_in,
        thinning=thinning,
    )

    metadata = dict(distribution.metadata)
    metadata.update(
        {
            "estimator": "occupation_measure",
            "stationarity_assumed": True,
            "stationarity_verified": False,
        }
    )

    return MonteCarloResult(
        method="empirical_stationary_distribution",
        estimate=distribution.estimate,
        states=distribution.states,
        n_samples=distribution.n_samples,
        n_paths=distribution.n_paths,
        burn_in=distribution.burn_in,
        thinning=distribution.thinning,
        metadata=metadata,
    )


# ============================================================================
# Hitting-event helpers
# ============================================================================


def _validate_target(
    target: State,
    *,
    states: tuple[State, ...],
    name: str = "target",
) -> State:
    """
    Validate that a target state belongs to the state space.
    """

    try:
        belongs = target in set(states)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be hashable."
        ) from exc

    if not belongs:
        raise ValueError(
            f"{name}={target!r} is not present in states."
        )

    return target


def _first_hitting_time(
    path: Path,
    *,
    target: State,
    include_initial: bool,
) -> int | None:
    """
    Return the first hitting time of a target or None.
    """

    start = 0 if include_initial else 1

    for time, state in enumerate(
        path[start:],
        start=start,
    ):
        if state == target:
            return time

    return None


# ============================================================================
# Empirical hitting probability
# ============================================================================


def empirical_hitting_probability(
    data: MonteCarloResult | Sequence[State],
    target: State,
    *,
    states: Sequence[State] | None = None,
    include_initial: bool = True,
) -> MonteCarloResult:
    """
    Estimate the probability that a path hits a target state.

    Each path contributes one Bernoulli observation:

    - one when the target is visited;
    - zero otherwise.

    For one raw path, the estimate is necessarily either zero or one.
    """

    paths = _extract_paths(data)
    frozen_states = _resolve_states(
        paths,
        states=states,
    )

    include_initial = _validate_boolean(
        include_initial,
        name="include_initial",
    )
    target = _validate_target(
        target,
        states=frozen_states,
    )

    indicators = np.array(
        [
            float(
                _first_hitting_time(
                    path,
                    target=target,
                    include_initial=include_initial,
                )
                is not None
            )
            for path in paths
        ],
        dtype=np.float64,
    )

    estimate = float(np.mean(indicators))
    variance = _sample_variance(indicators)
    standard_error = _standard_error(
        variance,
        n_samples=indicators.size,
    )

    hits = int(indicators.sum())

    return MonteCarloResult(
        method="empirical_hitting_probability",
        estimate=estimate,
        samples=_readonly_float_array(indicators),
        states=frozen_states,
        variance=variance,
        standard_error=standard_error,
        n_samples=indicators.size,
        n_paths=len(paths),
        metadata={
            "target": target,
            "include_initial": include_initial,
            "hits": hits,
            "misses": len(paths) - hits,
            "estimator": "bernoulli_path_average",
        },
    )


# ============================================================================
# Empirical hitting time
# ============================================================================


def empirical_hitting_time(
    data: MonteCarloResult | Sequence[State],
    target: State,
    *,
    states: Sequence[State] | None = None,
    include_initial: bool = True,
) -> MonteCarloResult:
    """
    Estimate the mean first hitting time of a target state.

    Only paths that hit the target contribute a finite hitting-time sample.
    Paths that do not hit the target within their observed horizon are
    recorded as censored in the result metadata.

    This omission policy is intentionally explicit. Formal treatment of
    right-censored observations belongs in a future survival-analysis layer.
    """

    paths = _extract_paths(data)
    frozen_states = _resolve_states(
        paths,
        states=states,
    )

    include_initial = _validate_boolean(
        include_initial,
        name="include_initial",
    )
    target = _validate_target(
        target,
        states=frozen_states,
    )

    hitting_times: list[float] = []
    censored = 0

    for path in paths:
        time = _first_hitting_time(
            path,
            target=target,
            include_initial=include_initial,
        )

        if time is None:
            censored += 1
        else:
            hitting_times.append(float(time))

    samples = _readonly_float_array(hitting_times)
    estimate = _mean_or_nan(samples)
    variance = _sample_variance(samples)
    standard_error = _standard_error(
        variance,
        n_samples=samples.size,
    )

    return MonteCarloResult(
        method="empirical_hitting_time",
        estimate=estimate,
        samples=samples,
        states=frozen_states,
        variance=variance,
        standard_error=standard_error,
        n_samples=samples.size,
        n_paths=len(paths),
        metadata={
            "target": target,
            "include_initial": include_initial,
            "successful_paths": samples.size,
            "censored_paths": censored,
            "censoring_policy": "omit",
            "estimator": "mean_observed_first_hitting_time",
        },
    )


# ============================================================================
# Empirical return time
# ============================================================================


def empirical_return_time(
    data: MonteCarloResult | Sequence[State],
    state: State,
    *,
    states: Sequence[State] | None = None,
) -> MonteCarloResult:
    """
    Estimate the mean recurrence time of a state.

    Every interval between two consecutive visits to ``state`` contributes
    one return-time observation. Intervals are collected independently within
    each path, so path boundaries never produce artificial returns.
    """

    paths = _extract_paths(data)
    frozen_states = _resolve_states(
        paths,
        states=states,
    )

    state = _validate_target(
        state,
        states=frozen_states,
        name="state",
    )

    return_times: list[float] = []
    paths_with_returns = 0

    for path in paths:
        visit_times = [
            time
            for time, observed_state in enumerate(path)
            if observed_state == state
        ]

        path_return_times = [
            float(right - left)
            for left, right in zip(
                visit_times[:-1],
                visit_times[1:],
            )
        ]

        if path_return_times:
            paths_with_returns += 1
            return_times.extend(path_return_times)

    samples = _readonly_float_array(return_times)
    estimate = _mean_or_nan(samples)
    variance = _sample_variance(samples)
    standard_error = _standard_error(
        variance,
        n_samples=samples.size,
    )

    return MonteCarloResult(
        method="empirical_return_time",
        estimate=estimate,
        samples=samples,
        states=frozen_states,
        variance=variance,
        standard_error=standard_error,
        n_samples=samples.size,
        n_paths=len(paths),
        metadata={
            "state": state,
            "paths_with_returns": paths_with_returns,
            "paths_without_returns": len(paths) - paths_with_returns,
            "estimator": "mean_intervisit_time",
        },
    )


__all__ = [
    "empirical_distribution",
    "empirical_transition_matrix",
    "empirical_stationary_distribution",
    "empirical_hitting_probability",
    "empirical_hitting_time",
    "empirical_return_time",
]