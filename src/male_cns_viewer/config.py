"""Load and validate the JSON workflow configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .data import SOURCES


@dataclass(frozen=True)
class DatasetConfig:
    regions: list[str]
    mip: int


@dataclass(frozen=True)
class NeuronConfig:
    body_ids: list[int]
    names: dict[str, str]
    show_skeletons: bool
    show_somas: bool
    soma_size_um: float


@dataclass(frozen=True)
class EmConfig:
    enabled: bool
    source: str
    detail_2d_mips: list[int]
    overview_3d_mip: int
    fast_2d_enabled: bool
    fast_2d_source: str
    fast_2d_mips: list[int]
    cache_gb: float
    visible_on_start: bool
    contrast_limits: list[float]


@dataclass(frozen=True)
class OutputConfig:
    directory: Path
    reuse_existing: bool
    shared_cache_directories: list[Path]


@dataclass(frozen=True)
class PerformanceConfig:
    n5_open_workers: int
    dask_workers: int
    show_timings: bool


@dataclass(frozen=True)
class NapariConfig:
    launch: bool
    ndisplay: int
    playback_step_slices: int
    neuropil_opacity: float
    em_opacity: float
    show_features_table: bool


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    neurons: NeuronConfig
    em: EmConfig
    output: OutputConfig
    performance: PerformanceConfig
    napari: NapariConfig


def load_config(path: Path) -> AppConfig:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    output_directory = Path(raw["output"]["directory"])
    if not output_directory.is_absolute():
        output_directory = path.parent / output_directory
    config = AppConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        neurons=NeuronConfig(**raw["neurons"]),
        em=EmConfig(**raw["em"]),
        output=OutputConfig(
            directory=output_directory,
            reuse_existing=raw["output"].get("reuse_existing", True),
            shared_cache_directories=[
                (item if item.is_absolute() else path.parent / item).resolve()
                for item in map(Path, raw["output"].get("shared_cache_directories", []))
            ],
        ),
        performance=PerformanceConfig(**raw["performance"]),
        napari=NapariConfig(**raw["napari"]),
    )
    unknown = set(config.dataset.regions) - set(SOURCES)
    if not config.dataset.regions or unknown:
        raise ValueError(f"Invalid dataset.regions: {config.dataset.regions}")
    if config.dataset.mip not in (2, 3):
        raise ValueError("dataset.mip must be 2 or 3 for this prototype")
    if not config.neurons.body_ids or len(set(config.neurons.body_ids)) != len(config.neurons.body_ids):
        raise ValueError("neurons.body_ids must contain unique IDs")
    if config.neurons.soma_size_um <= 0:
        raise ValueError("neurons.soma_size_um must be positive")
    if config.em.enabled:
        detail_mips = config.em.detail_2d_mips
        if not detail_mips or detail_mips != sorted(set(detail_mips)):
            raise ValueError("em.detail_2d_mips must be sorted and contain no duplicates")
        if any(mip not in range(10) for mip in detail_mips):
            raise ValueError("N5 em.detail_2d_mips must be between 0 and 9")
        if config.em.overview_3d_mip not in range(10):
            raise ValueError("N5 em.overview_3d_mip must be between 0 and 9")
        if config.em.fast_2d_enabled:
            fast_mips = config.em.fast_2d_mips
            if not fast_mips or fast_mips != sorted(set(fast_mips)):
                raise ValueError("em.fast_2d_mips must be sorted and contain no duplicates")
            if any(mip not in range(11) for mip in fast_mips):
                raise ValueError("em.fast_2d_mips must be between 0 and 10")
        if config.em.cache_gb <= 0:
            raise ValueError("em.cache_gb must be positive")
        if len(config.em.contrast_limits) != 2 or config.em.contrast_limits[0] >= config.em.contrast_limits[1]:
            raise ValueError("em.contrast_limits must be [minimum, maximum]")
    if config.napari.ndisplay not in (2, 3):
        raise ValueError("napari.ndisplay must be 2 or 3")
    if config.napari.playback_step_slices <= 0:
        raise ValueError("napari.playback_step_slices must be a positive integer")
    for opacity in (config.napari.neuropil_opacity, config.napari.em_opacity):
        if not 0 <= opacity <= 1:
            raise ValueError("napari opacities must be between 0 and 1")
    if config.performance.n5_open_workers <= 0 or config.performance.dask_workers <= 0:
        raise ValueError("performance worker counts must be positive")
    return config
