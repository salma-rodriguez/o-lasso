"""
spectral_operators.io
=====================

Input/output utilities for spectral operator objects.

This module provides JSON serialization, operator persistence,
diagnostic-report storage, and NumPy array input/output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.algebra import LinearOperator
from .core.exceptions import (
    OperatorError,
    SerializationError,
)


# ===========================================================================
# Shared Helpers
# ===========================================================================

def _as_path(path) -> Path:
    """
    Convert a path-like value into a Path object.
    """

    try:
        return Path(path)
    except TypeError as exc:
        raise SerializationError(
            "path must be a valid path-like object."
        ) from exc


# ===========================================================================
# JSON Utilities
# ===========================================================================

class JSONSerializer:
    """
    Recursive JSON serialization utilities.

    Complex values are represented as dictionaries of the form

        {
            "__complex__": True,
            "real": ...,
            "imag": ...
        }
    """

    @staticmethod
    def from_jsonable(obj):
        """
        Recursively reconstruct supported Python and NumPy values.
        """

        if isinstance(obj, list):
            return [
                JSONSerializer.from_jsonable(value)
                for value in obj
            ]

        if isinstance(obj, dict):
            if obj.get("__complex__") is True:
                try:
                    return complex(
                        obj["real"],
                        obj["imag"],
                    )
                except KeyError as exc:
                    raise SerializationError(
                        "Malformed complex-number encoding."
                    ) from exc

            return {
                key: JSONSerializer.from_jsonable(value)
                for key, value in obj.items()
            }

        return obj

    @staticmethod
    def to_jsonable(obj):
        """
        Recursively convert supported objects into JSON-safe values.
        """

        if isinstance(obj, np.ndarray):
            return JSONSerializer.to_jsonable(
                obj.tolist()
            )

        if isinstance(obj, np.generic):
            return JSONSerializer.to_jsonable(
                obj.item()
            )

        if isinstance(obj, complex):
            return {
                "__complex__": True,
                "real": float(obj.real),
                "imag": float(obj.imag),
            }

        if isinstance(obj, dict):
            return {
                str(key): JSONSerializer.to_jsonable(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [
                JSONSerializer.to_jsonable(value)
                for value in obj
            ]

        if isinstance(obj, Path):
            return str(obj)

        return obj

    @staticmethod
    def dump(
        obj,
        path,
        *,
        indent: int | None = 2,
        **kwargs,
    ) -> None:
        """
        Serialize an object to a JSON file.
        """

        destination = _as_path(path)

        try:
            with destination.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    JSONSerializer.to_jsonable(obj),
                    file,
                    indent=indent,
                    **kwargs,
                )
        except (OSError, TypeError, ValueError) as exc:
            raise SerializationError(
                f"Could not write JSON data to {destination}."
            ) from exc

    @staticmethod
    def dumps(
        obj,
        **kwargs,
    ) -> str:
        """
        Serialize an object to a JSON string.
        """

        try:
            return json.dumps(
                JSONSerializer.to_jsonable(obj),
                **kwargs,
            )
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                "Could not serialize object to JSON."
            ) from exc

    @staticmethod
    def load(path):
        """
        Load and decode a JSON file.
        """

        source = _as_path(path)

        try:
            with source.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SerializationError(
                f"Could not read JSON data from {source}."
            ) from exc

        return JSONSerializer.from_jsonable(
            data
        )

    @staticmethod
    def loads(value: str):
        """
        Load and decode a JSON string.
        """

        if not isinstance(value, str):
            raise SerializationError(
                "value must be a JSON string."
            )

        try:
            data = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SerializationError(
                "Could not decode JSON string."
            ) from exc

        return JSONSerializer.from_jsonable(
            data
        )


# ===========================================================================
# Operator IO
# ===========================================================================

class OperatorIO:
    """
    Save and load LinearOperator objects.

    Notes
    -----
    Deserialization reconstructs the base LinearOperator class. It does
    not currently reconstruct concrete subclasses such as
    FiniteDifferenceOperator or ZetaOperator.
    """

    @staticmethod
    def from_dict(
        data: dict,
    ) -> LinearOperator:
        """
        Construct a LinearOperator from serialized data.
        """

        if not isinstance(data, dict):
            raise OperatorError(
                "data must be a dictionary."
            )

        if "matrix" not in data:
            raise SerializationError(
                "Serialized operator data must contain 'matrix'."
            )

        try:
            matrix = np.asarray(
                data["matrix"]
            )
        except (TypeError, ValueError) as exc:
            raise SerializationError(
                "Serialized matrix data is invalid."
            ) from exc

        metadata = data.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            raise SerializationError(
                "Serialized metadata must be a dictionary."
            )

        return LinearOperator(
            matrix=matrix,
            name=data.get(
                "name",
                "LinearOperator",
            ),
            metadata=metadata,
        )

    @staticmethod
    def load_json(
        path,
    ) -> LinearOperator:
        """
        Load a LinearOperator from JSON.
        """

        return OperatorIO.from_dict(
            JSONSerializer.load(path)
        )

    @staticmethod
    def save_json(
        operator: LinearOperator,
        path,
    ) -> None:
        """
        Save a LinearOperator to JSON.
        """

        JSONSerializer.dump(
            OperatorIO.to_dict(operator),
            path,
        )

    @staticmethod
    def to_dict(
        operator: LinearOperator,
    ) -> dict:
        """
        Convert a LinearOperator into serializable data.
        """

        if not isinstance(operator, LinearOperator):
            raise OperatorError(
                "operator must be a LinearOperator."
            )

        return {
            "name": operator.name,
            "matrix": JSONSerializer.to_jsonable(
                operator.matrix
            ),
            "metadata": JSONSerializer.to_jsonable(
                operator.metadata
            ),
        }


# ===========================================================================
# Diagnostic IO
# ===========================================================================

class DiagnosticIO:
    """
    Save and load diagnostic dictionaries and reports.
    """

    @staticmethod
    def load(
        path,
    ) -> dict:
        """
        Load a diagnostic report.
        """

        report = JSONSerializer.load(path)

        if not isinstance(report, dict):
            raise SerializationError(
                "Diagnostic data must decode to a dictionary."
            )

        return report

    @staticmethod
    def save(
        report: dict,
        path,
    ) -> None:
        """
        Save a diagnostic report.
        """

        if not isinstance(report, dict):
            raise OperatorError(
                "report must be a dictionary."
            )

        JSONSerializer.dump(
            report,
            path,
        )


# ===========================================================================
# NumPy IO
# ===========================================================================

class NumpyIO:
    """
    Save and load NumPy array data.
    """

    @staticmethod
    def load_array(
        path,
    ) -> np.ndarray:
        """
        Load an array from an NPY file.
        """

        source = _as_path(path)

        try:
            return np.load(
                source,
                allow_pickle=False,
            )
        except (OSError, ValueError) as exc:
            raise SerializationError(
                f"Could not load NumPy array from {source}."
            ) from exc

    @staticmethod
    def load_npz(
        path,
    ) -> dict[str, np.ndarray]:
        """
        Load arrays from an NPZ archive.
        """

        source = _as_path(path)

        try:
            with np.load(
                source,
                allow_pickle=False,
            ) as data:
                return {
                    key: np.array(
                        data[key],
                        copy=True,
                    )
                    for key in data.files
                }
        except (OSError, ValueError) as exc:
            raise SerializationError(
                f"Could not load NumPy archive from {source}."
            ) from exc

    @staticmethod
    def save_array(
        array,
        path,
    ) -> None:
        """
        Save an array in NPY format.
        """

        destination = _as_path(path)

        try:
            np.save(
                destination,
                np.asarray(array),
                allow_pickle=False,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SerializationError(
                f"Could not save NumPy array to {destination}."
            ) from exc

    @staticmethod
    def save_npz(
        path,
        **arrays,
    ) -> None:
        """
        Save named arrays in an NPZ archive.
        """

        if not arrays:
            raise OperatorError(
                "At least one named array is required."
            )

        destination = _as_path(path)

        converted = {
            key: np.asarray(value)
            for key, value in arrays.items()
        }

        try:
            np.savez(
                destination,
                **converted,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SerializationError(
                f"Could not save NumPy archive to {destination}."
            ) from exc
