"""Helpers for building a trial-aligned dataset from an NWB file."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from pynwb import NWBHDF5IO
from pynwb.base import TimeSeries
from pynwb.ecephys import ElectricalSeries, SpikeEventSeries


def load_nwb(path: str):
    """Open an NWB file from disk and return the in-memory NWBFile object.

    This keeps file validation in one place so the training script can fail
    early with a readable error when the path is missing or unreadable.
    """
    if not path:
        raise ValueError("An NWB path is required when using NWB mode.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NWB file not found: {path}")

    io = NWBHDF5IO(path, "r")
    nwb = io.read()
    # Keep a reference to the live reader because DynamicTable columns can be
    # backed by lazy HDF5 datasets that still need the file handle.
    nwb._cached_io = io
    return nwb


def _get_trials_table(nwb: Any):
    """Return the trials DynamicTable and validate the required columns."""
    if getattr(nwb, "intervals", None) is None or "trials" not in nwb.intervals:
        raise ValueError("NWB file does not contain nwb.intervals['trials'].")

    trials = nwb.intervals["trials"]
    required_columns = {"start_time", "stop_time", "index_target_position"}
    missing = sorted(required_columns.difference(trials.colnames))
    if missing:
        raise ValueError(
            "NWB trials table is missing required columns: "
            + ", ".join(missing)
        )

    return trials


def inspect_nwb_neural_objects(nwb) -> list[dict[str, Any]]:
    """Inspect the NWB file and collect time-varying neural data candidates.

    The search walks through the requested NWB containers as well as
    ``nwb.objects`` because some files expose the actual series there even when
    top-level acquisition/processing collections are empty.
    """
    candidates = []
    seen_object_ids = set()

    def register(obj: Any, location: str):
        object_id = getattr(obj, "object_id", None)
        if object_id in seen_object_ids:
            return
        if not isinstance(obj, (TimeSeries, ElectricalSeries, SpikeEventSeries)):
            return

        data = getattr(obj, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None or len(shape) == 0:
            return

        name = getattr(obj, "name", type(obj).__name__)
        lowered_name = str(name).lower()
        is_neural = isinstance(obj, (ElectricalSeries, SpikeEventSeries)) or any(
            token in lowered_name
            for token in ["spike", "spiking", "threshold", "lfp", "neural", "electrical"]
        )
        if not is_neural:
            return

        num_channels = int(shape[1]) if len(shape) > 1 else 1
        priority = 100
        if "spikingbandpower" in lowered_name:
            priority = 0
        elif "thresholdcrossings" in lowered_name:
            priority = 1
        elif isinstance(obj, ElectricalSeries):
            priority = 2
        elif isinstance(obj, SpikeEventSeries):
            priority = 3
        else:
            priority = 10

        candidates.append(
            {
                "name": name,
                "type": type(obj).__name__,
                "location": location,
                "shape": tuple(int(dim) for dim in shape),
                "num_channels": num_channels,
                "priority": priority,
                "object": obj,
            }
        )
        seen_object_ids.add(object_id)

    for name, obj in getattr(nwb, "analysis", {}).items():
        register(obj, f"analysis/{name}")

    for name, obj in getattr(nwb, "stimulus", {}).items():
        register(obj, f"stimulus/{name}")

    for name, obj in getattr(nwb, "processing", {}).items():
        register(obj, f"processing/{name}")
        for interface_name, interface_obj in getattr(obj, "data_interfaces", {}).items():
            register(interface_obj, f"processing/{name}/{interface_name}")

    for name, obj in getattr(nwb, "acquisition", {}).items():
        register(obj, f"acquisition/{name}")

    for object_id, obj in getattr(nwb, "objects", {}).items():
        register(obj, f"objects/{object_id}")

    candidates.sort(
        key=lambda item: (
            item["priority"],
            -item["num_channels"],
            item["name"].lower(),
        )
    )
    return candidates


def _get_series_timestamps(series) -> np.ndarray:
    """Return timestamps for a series using explicit timestamps or rate metadata."""
    if getattr(series, "timestamps", None) is not None:
        return np.asarray(series.timestamps[:], dtype=np.float64)

    rate = getattr(series, "rate", None)
    starting_time = getattr(series, "starting_time", None)
    if rate is None or starting_time is None:
        raise ValueError(
            f"Series {getattr(series, 'name', '<unnamed>')} has neither timestamps nor rate metadata."
        )

    num_samples = int(series.data.shape[0])
    return starting_time + np.arange(num_samples, dtype=np.float64) / float(rate)


def _read_series_data(series) -> np.ndarray:
    """Load a neural series into a NumPy array with samples on axis 0."""
    data = np.asarray(series.data[:], dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D neural data for {getattr(series, 'name', '<unnamed>')}, got shape {data.shape}."
        )
    return data


def extract_trial_labels(nwb) -> np.ndarray:
    """Build the label vector from trial target positions.

    The requested NWB column, ``index_target_position``, can be stored as
    floating-point target positions rather than integer class IDs. To keep the
    existing classification pipeline working, this function maps each unique
    target position to a contiguous integer label.
    """
    trials = _get_trials_table(nwb)
    raw_labels = np.asarray(trials["index_target_position"].data[:])
    if raw_labels.size == 0:
        raise ValueError("The NWB trials table is empty.")

    unique_labels, encoded_labels = np.unique(raw_labels, return_inverse=True)
    if unique_labels.size < 2:
        raise ValueError(
            "Need at least two distinct index_target_position values to train."
        )

    return encoded_labels.astype(np.int64)


def build_trial_features(nwb) -> np.ndarray:
    """Create a simple numeric feature matrix from the trials table.

    This baseline intentionally uses trial-level metadata only. It always
    includes trial duration and then appends additional scalar numeric trial
    columns when they are present and safe to use. The label column and object
    columns such as ``target_style`` are excluded to avoid leakage and keep the
    feature matrix numeric.
    """
    trials = _get_trials_table(nwb)

    start_time = np.asarray(trials["start_time"].data[:], dtype=np.float32)
    stop_time = np.asarray(trials["stop_time"].data[:], dtype=np.float32)
    if start_time.shape != stop_time.shape:
        raise ValueError("start_time and stop_time columns do not align.")

    duration = stop_time - start_time
    if np.any(~np.isfinite(duration)):
        raise ValueError("Non-finite values found while computing trial duration.")

    feature_columns = [duration.astype(np.float32)]

    excluded_columns = {
        "start_time",
        "stop_time",
        "index_target_position",
        "mrs_target_position",
        "target_style",
        "timeseries",
    }

    preferred_numeric_columns = [
        "trial_number",
        "trial_count",
        "run_id",
        "trial_timeout",
    ]

    for column_name in preferred_numeric_columns:
        if column_name not in trials.colnames or column_name in excluded_columns:
            continue

        column_values = np.asarray(trials[column_name].data[:])
        if column_values.ndim != 1:
            continue
        if not np.issubdtype(column_values.dtype, np.number):
            continue

        numeric_values = column_values.astype(np.float32)
        if np.any(~np.isfinite(numeric_values)):
            continue

        feature_columns.append(numeric_values)

    if not feature_columns:
        raise ValueError("No numeric trial-level features could be built from NWB.")

    return np.column_stack(feature_columns).astype(np.float32)


def extract_neural_features_from_nwb(nwb) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Extract trial-aligned neural features from the best available NWB series.

    This function searches the NWB file for time-varying neural data, prefers
    ``SpikingBandPower`` when present, aligns the selected series to each trial
    window, and computes simple per-trial summary features per channel:
    mean, standard deviation, and energy.
    """
    candidates = inspect_nwb_neural_objects(nwb)
    if not candidates:
        return None, {
            "used_neural_features": False,
            "reason": "No time-varying neural series found in NWB.",
            "candidates": [],
        }

    selected = candidates[0]
    series = selected["object"]
    data = _read_series_data(series)
    timestamps = _get_series_timestamps(series)
    if data.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"Series {selected['name']} data/timestamp length mismatch: "
            f"{data.shape[0]} vs {timestamps.shape[0]}."
        )

    trials = _get_trials_table(nwb)
    start_times = np.asarray(trials["start_time"].data[:], dtype=np.float64)
    stop_times = np.asarray(trials["stop_time"].data[:], dtype=np.float64)
    if start_times.shape != stop_times.shape:
        raise ValueError("start_time and stop_time columns do not align.")

    feature_rows = []
    window_lengths = []
    for start_time, stop_time in zip(start_times, stop_times):
        left = int(np.searchsorted(timestamps, start_time, side="left"))
        right = int(np.searchsorted(timestamps, stop_time, side="right"))
        window = data[left:right]

        if window.shape[0] == 0:
            # If a trial falls between sampled timestamps, use the nearest sample
            # so the pipeline still produces one feature vector per trial.
            nearest_index = int(np.clip(np.searchsorted(timestamps, start_time), 0, len(timestamps) - 1))
            window = data[nearest_index:nearest_index + 1]

        window_lengths.append(int(window.shape[0]))
        mean_per_channel = window.mean(axis=0)
        std_per_channel = window.std(axis=0)
        energy_per_channel = np.sum(window ** 2, axis=0)
        feature_rows.append(
            np.concatenate(
                [mean_per_channel, std_per_channel, energy_per_channel],
                axis=0,
            )
        )

    X = np.stack(feature_rows, axis=0).astype(np.float32)
    return X, {
        "used_neural_features": True,
        "selected_name": selected["name"],
        "selected_type": selected["type"],
        "selected_location": selected["location"],
        "signal_shape": tuple(int(dim) for dim in data.shape),
        "num_channels": int(data.shape[1]),
        "num_trials": int(X.shape[0]),
        "window_length_min": int(np.min(window_lengths)),
        "window_length_median": float(np.median(window_lengths)),
        "window_length_max": int(np.max(window_lengths)),
        "final_feature_shape": tuple(int(dim) for dim in X.shape),
        "candidates": [
            {
                "name": item["name"],
                "type": item["type"],
                "location": item["location"],
                "shape": item["shape"],
            }
            for item in candidates
        ],
    }


def get_dataset_from_nwb(
    path: str,
    feature_mode: str = "auto",
    return_info: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one NWB file and return ``(X, y)`` as NumPy arrays.

    ``X`` is a 2D trial-feature matrix of shape ``[num_trials, num_features]``.
    ``y`` is a 1D integer label vector aligned to the same trial order.
    """
    if feature_mode not in {"auto", "neural", "metadata", "combined"}:
        raise ValueError(
            "feature_mode must be one of: auto, neural, metadata, combined."
        )

    nwb = load_nwb(path)
    try:
        metadata_X = build_trial_features(nwb)
        neural_X, neural_info = extract_neural_features_from_nwb(nwb)

        if feature_mode == "metadata":
            X = metadata_X
            selected_feature_mode = "metadata"
        elif feature_mode == "combined":
            if neural_X is None:
                raise ValueError(
                    "Requested combined NWB features, but no neural series was found."
                )
            X = np.concatenate([neural_X, metadata_X], axis=1).astype(np.float32)
            selected_feature_mode = "combined"
        elif feature_mode == "neural":
            if neural_X is None:
                raise ValueError(
                    "Requested neural NWB features, but no neural series was found."
                )
            X = neural_X
            selected_feature_mode = "neural"
        else:
            if neural_X is not None:
                X = neural_X
                selected_feature_mode = "neural"
            else:
                X = metadata_X
                selected_feature_mode = "metadata_fallback"

        y = extract_trial_labels(nwb)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Feature/label size mismatch from NWB: X has {X.shape[0]} rows, "
                f"y has {y.shape[0]} labels."
            )

        info = {
            "requested_feature_mode": feature_mode,
            "selected_feature_mode": selected_feature_mode,
            "metadata_feature_shape": tuple(int(dim) for dim in metadata_X.shape),
            "neural_feature_shape": None if neural_X is None else tuple(int(dim) for dim in neural_X.shape),
            "final_feature_shape": tuple(int(dim) for dim in X.shape),
            "num_trials": int(y.shape[0]),
            "neural_info": neural_info,
        }

        if return_info:
            return X.astype(np.float32), y.astype(np.int64), info
        return X.astype(np.float32), y.astype(np.int64)
    finally:
        cached_io = getattr(nwb, "_cached_io", None)
        if cached_io is not None:
            cached_io.close()
