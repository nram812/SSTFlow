"""Shared constants, configuration loading, date selection, and atomic I/O.

Everything in this package assumes a single scalar target variable (sea-surface
temperature in degrees Celsius) on a fixed regular latitude/longitude grid whose
land cells are permanently missing.  The land mask is therefore *static* and is
computed once from the first time step during preprocessing.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

#: Name of the target variable inside the raw OFAM NetCDF file.
SOURCE_VARIABLE = "temp"

#: Dimension names inside the raw OFAM NetCDF file.
SOURCE_TIME_DIM = "Time"
SOURCE_DEPTH_DIM = "st_ocean"
SOURCE_LAT_DIM = "yt_ocean"
SOURCE_LON_DIM = "xt_ocean"

#: Public name used for the variable throughout the derived products.
TARGET_NAME = "sst"

#: Value substituted for land (and for invalid coarse cells) *after* the
#: normalisation step.  Zero is the mean of the normalised ocean distribution,
#: so land pixels carry no gradient signal and cannot blow up any activation.
LAND_FILL_VALUE = 0.0

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Keys in a config whose values are paths resolved relative to the repository.
_RELATIVE_PATH_KEYS = (
    "derived_path",
    "normalization_cache",
    "output_dir",
    "smoke_output_dir",
    "resume_from",
)

_RELATIVE_PATH_LIST_KEYS = ("source_paths",)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_config(path: str | Path) -> dict:
    """Load a JSON config and resolve repository-relative paths to absolutes."""
    config = load_json(path)
    for key in _RELATIVE_PATH_KEYS:
        if key not in config:
            continue
        value = Path(config[key])
        config[key] = str(
            value if value.is_absolute() else REPOSITORY_ROOT / value
        )
    for key in _RELATIVE_PATH_LIST_KEYS:
        if key not in config:
            continue
        config[key] = [
            str(value if (value := Path(item)).is_absolute() else REPOSITORY_ROOT / value)
            for item in config[key]
        ]
    config.setdefault("config_path", str(Path(path).resolve()))
    return config


def atomic_json(path: str | Path, payload) -> None:
    """Write JSON via a temporary file so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def atomic_torch_save(payload, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------
def date_keys(values) -> np.ndarray:
    """Return ``YYYY-MM-DD`` strings for any array of datetime-like values."""
    return np.asarray([str(value)[:10] for value in values])


def selected_indices(time_values, ranges: list[list[str]]) -> np.ndarray:
    """Return the indices of ``time_values`` covered by inclusive date ranges."""
    keys = date_keys(time_values)
    selected = np.zeros(len(keys), dtype=bool)
    for start, stop in ranges:
        if start > stop:
            raise ValueError(f"Invalid date range {start} to {stop}")
        selected |= (keys >= start) & (keys <= stop)
    result = np.flatnonzero(selected)
    if not len(result):
        raise ValueError(f"No dates selected by {ranges}")
    return result


def contiguous_runs(indices: np.ndarray):
    """Yield ``(source_start, source_stop, destination)`` half-open runs.

    Consecutive indices are grouped so that a strided NetCDF/HDF5 file can be
    read with a small number of large slices instead of many scalar reads.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return
    start = previous = int(indices[0])
    destination = 0
    for raw_value in indices[1:]:
        value = int(raw_value)
        if value != previous + 1:
            yield start, previous + 1, destination
            destination += previous + 1 - start
            start = value
        previous = value
    yield start, previous + 1, destination


def consecutive_pair_starts(
    time_values, ranges: list[list[str]], lag: int = 1
) -> np.ndarray:
    """Indices ``t`` where ``t`` and ``t + lag`` are both inside one range.

    Used by the autoregressive dataset so a training pair never straddles the
    boundary between the training and validation periods.
    """
    if lag < 1:
        raise ValueError("lag must be positive")
    starts: list[int] = []
    for date_range in ranges:
        indices = selected_indices(time_values, [date_range])
        if len(indices) <= lag:
            raise ValueError(
                f"Range {date_range} has {len(indices)} days, too short for lag {lag}"
            )
        days = np.asarray(
            date_keys(time_values)[indices], dtype="datetime64[D]"
        )
        gaps = np.diff(days) != np.timedelta64(1, "D")
        if gaps.any():
            raise ValueError(
                f"Non-consecutive dates inside range {date_range} at "
                f"{days[np.flatnonzero(gaps)[0]]}"
            )
        starts.extend(int(value) for value in indices[:-lag])
    return np.asarray(sorted(starts), dtype=np.int64)


# --------------------------------------------------------------------------
# hashing / reproducibility
# --------------------------------------------------------------------------
def mask_sha256(mask, lat, lon) -> str:
    """Fingerprint a land/ocean mask together with the grid it belongs to."""
    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
    digest.update(np.asarray(lat, dtype=np.float64).tobytes())
    digest.update(np.asarray(lon, dtype=np.float64).tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_payload() -> dict:
    payload = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(state: dict) -> None:
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"].cpu())
    if "cuda_rng_state" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [
                tensor.cpu()
                for tensor in state["cuda_rng_state"][: torch.cuda.device_count()]
            ]
        )


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------
def normalize(values, mean: float, std: float):
    """Standardise physical values; works for numpy arrays and torch tensors."""
    return (values - mean) / std


def denormalize(values, mean: float, std: float):
    return values * std + mean


def normalized_to_physical(
    values: torch.Tensor, normalization: dict
) -> torch.Tensor:
    """Undo standardisation and re-insert NaN over land.

    ``values`` has shape ``(sample, channel, lat, lon)``.  The returned tensor is
    in degrees Celsius with land set to NaN so that downstream ``nanmean``-style
    metrics and NetCDF products behave exactly like the source data.
    """
    mean = float(normalization["sst_mean"])
    std = float(normalization["sst_std"])
    physical = values * std + mean
    mask = normalization.get("_ocean_mask_tensor")
    if mask is not None:
        mask = mask.to(device=physical.device)
        physical = torch.where(
            mask.bool().expand_as(physical),
            physical,
            torch.full_like(physical, float("nan")),
        )
    return physical


def attach_ocean_mask(normalization: dict, ocean_mask: np.ndarray) -> dict:
    """Store the high-resolution ocean mask on a normalisation dictionary."""
    normalization = dict(normalization)
    normalization["_ocean_mask_tensor"] = torch.from_numpy(
        np.asarray(ocean_mask, dtype=bool)[None, None]
    )
    return normalization


def json_safe(normalization: dict) -> dict:
    """Drop private tensor entries so a normalisation dict can be serialised."""
    return {
        key: value
        for key, value in normalization.items()
        if not key.startswith("_")
    }
