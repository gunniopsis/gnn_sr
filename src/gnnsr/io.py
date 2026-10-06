from __future__ import annotations

import csv
import struct
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

import numpy as np


@dataclass(frozen=True)
class NpyMemberInfo:
    dtype: np.dtype
    shape: tuple[int, ...]
    fortran_order: bool
    data_offset: int


def _member_name(key: str) -> str:
    return key if key.endswith(".npy") else f"{key}.npy"


def npz_member_info(path: str | Path, key: str) -> NpyMemberInfo:
    """Return the on-disk layout of an uncompressed NPY member inside an NPZ file."""
    path = Path(path)
    member = _member_name(key)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{path}:{member} is compressed and cannot be memory-mapped. "
                "Create an uncompressed cache with numpy.save or numpy.savez."
            )
        with path.open("rb") as raw:
            raw.seek(info.header_offset)
            local_header = raw.read(30)
            if len(local_header) != 30:
                raise ValueError(f"Invalid local ZIP header in {path}")
            signature, *_, filename_length, extra_length = struct.unpack(
                "<IHHHHHIIIHH", local_header
            )
            if signature != 0x04034B50:
                raise ValueError(f"Invalid ZIP signature in {path}")
            npy_start = info.header_offset + 30 + filename_length + extra_length
            raw.seek(npy_start)
            version = np.lib.format.read_magic(raw)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(raw)
            elif version in ((2, 0), (3, 0)):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(raw)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}:{member}")
            return NpyMemberInfo(np.dtype(dtype), tuple(shape), fortran_order, raw.tell())


def memory_map_npz(path: str | Path, key: str, mode: str = "r") -> np.memmap:
    """Memory-map an uncompressed array stored inside an NPZ without extracting it."""
    path = Path(path)
    info = npz_member_info(path, key)
    order = "F" if info.fortran_order else "C"
    return np.memmap(
        path,
        dtype=info.dtype,
        mode=mode,
        offset=info.data_offset,
        shape=info.shape,
        order=order,
    )


@dataclass(frozen=True)
class EventRecord:
    index: int
    name: str
    group: int
    hec_plan: str
    tuflow_path: Path
    hec_path: Path


class EventCatalog:
    """Burnett River event metadata in the exact order used by the LSG scripts."""

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        summary = self.root / "BurnettRV_event_summary.csv"
        high_root = self.root / "HD_model_data" / "High-fidelity"
        low_root = self.root / "HD_model_data" / "Low-fidelity"
        records: list[EventRecord] = []
        with summary.open("r", newline="", encoding="utf-8-sig") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                name = row["tuflow_evt_name_old"]
                plan = row["HEC_RAS_plan"]
                tuflow = high_root / f"Paradise_{name}_002.npz"
                hec = low_root / f"BurnettRV_LFmodelB.{plan}.hdf"
                if not tuflow.exists():
                    raise FileNotFoundError(tuflow)
                if not hec.exists():
                    raise FileNotFoundError(hec)
                records.append(EventRecord(index, name, int(row["Group"]), plan, tuflow, hec))
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self.records)

    def training_events(self, held_out_group: int) -> tuple[EventRecord, ...]:
        return tuple(record for record in self.records if record.group != held_out_group)

    def test_events(self, held_out_group: int) -> tuple[EventRecord, ...]:
        return tuple(record for record in self.records if record.group == held_out_group)


class TuflowEvent:
    """Lazy access to one high-fidelity event."""

    def __init__(self, path: str | Path, warmup_steps: int = 48):
        self.path = Path(path)
        self.warmup_steps = warmup_steps
        self._wse = memory_map_npz(self.path, "wl_data")
        with np.load(self.path) as archive:
            self.time = np.asarray(archive["time_data"])[warmup_steps:]
        if self._wse.shape[0] < warmup_steps:
            raise ValueError(f"Warm-up exceeds event length for {self.path}")

    @property
    def n_timesteps(self) -> int:
        return self._wse.shape[0] - self.warmup_steps

    @property
    def n_cells(self) -> int:
        return self._wse.shape[1]

    def read_wse(self, timestep: int, cell_indices: np.ndarray) -> np.ndarray:
        if not 0 <= timestep < self.n_timesteps:
            raise IndexError(timestep)
        row = self._wse[self.warmup_steps + timestep]
        return np.asarray(row[cell_indices], dtype=np.float32)


def _decode_hec_times(values: np.ndarray) -> np.ndarray:
    parsed = [
        datetime.strptime(  # noqa: DTZ007 - HEC timestamps are local and only differences are used
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value),
            "%d%b%Y %H:%M:%S",
        )
        for value in values
    ]
    return np.asarray(parsed, dtype="datetime64[s]")


class HecEvent:
    """Lazy access to selected non-ghost HEC-RAS cells in one event HDF file."""

    def __init__(
        self,
        path: str | Path,
        flow_area_name: str,
        compact_cell_indices: np.ndarray,
        warmup_hours: float = 47.5,
    ):
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific message
            raise ImportError("Reading HEC-RAS results requires h5py") from exc
        self.path = Path(path)
        self._handle = h5py.File(self.path, "r")
        geometry = self._handle[f"Geometry/2D Flow Areas/{flow_area_name}"]
        raw_elevation = np.asarray(geometry["Cells Minimum Elevation"])
        valid_raw = np.flatnonzero(np.isfinite(raw_elevation))
        compact_cell_indices = np.asarray(compact_cell_indices, dtype=np.int64)
        self._raw_indices = valid_raw[compact_cell_indices]
        base = self._handle[
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
        ]
        self._wse = base[f"2D Flow Areas/{flow_area_name}/Water Surface"]
        times = _decode_hec_times(np.asarray(base["Time Date Stamp"][:]))
        threshold = times[0] + np.timedelta64(round(warmup_hours * 3600), "s")
        self._kept_timesteps = np.flatnonzero(times > threshold)

    @property
    def n_timesteps(self) -> int:
        return len(self._kept_timesteps)

    def read_wse(self, timestep: int) -> np.ndarray:
        if not 0 <= timestep < self.n_timesteps:
            raise IndexError(timestep)
        raw_timestep = int(self._kept_timesteps[timestep])
        values = np.asarray(self._wse[raw_timestep, self._raw_indices], dtype=np.float32)
        return np.round(values, 3)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


def event_timestep_counts(
    events: tuple[EventRecord, ...] | list[EventRecord], warmup_steps: int
) -> np.ndarray:
    counts = []
    for event in events:
        info = npz_member_info(event.tuflow_path, "wl_data")
        counts.append(info.shape[0] - warmup_steps)
    return np.asarray(counts, dtype=np.int64)


def flatten_event_timesteps(
    events: tuple[EventRecord, ...] | list[EventRecord], warmup_steps: int
) -> list[tuple[int, int]]:
    counts = event_timestep_counts(events, warmup_steps)
    return [
        (event.index, timestep)
        for event, count in zip(events, counts, strict=True)
        for timestep in range(int(count))
    ]
