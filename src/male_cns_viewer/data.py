"""Read official Male CNS neuropil labels from Janelia."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import requests
from cloudvolume import CloudVolume
from cloudvolume.datasource.precomputed.image import rx
from cloudfiles.monitoring import TransmissionMonitor

from .colors import neuropil_color, split_side


SOURCES = {
    "brain": "gs://flyem-male-cns/rois/fullbrain-roi-v4",
    "vnc": "gs://flyem-male-cns/rois/malecns-vnc-neuropil-roi-v0",
}


def patch_cloudfiles_windows_timer() -> None:
    """Prevent zero-length telemetry intervals on coarse Windows clocks."""
    if getattr(TransmissionMonitor, "_male_cns_timer_patch", False):
        return

    def end_io(self, flight_id, num_bytes):
        end_us = int(time.monotonic() * 1e6)
        with self._lock:
            start_us = int(self._in_flight.pop(flight_id) * 1e6)
            self._in_flight_bytes -= num_bytes
            self._intervaltree.addi(start_us, max(end_us, start_us + 1), [flight_id, num_bytes])
            self._total_bytes_landed += num_bytes

    TransmissionMonitor.end_io = end_io
    TransmissionMonitor._male_cns_timer_patch = True


@dataclass(frozen=True)
class LabelVolume:
    dataset: str
    source: str
    mip: int
    labels_xyz: np.ndarray
    spacing_xyz_um: np.ndarray
    origin_xyz_um: np.ndarray


@dataclass(frozen=True)
class RegionInfo:
    label_id: int
    name: str
    side: str
    color: str


def load_label_volume(dataset: str, mip: int) -> LabelVolume:
    patch_cloudfiles_windows_timer()
    rx.DEFAULT_THREADS = 1
    source = SOURCES[dataset]
    cloud_volume = CloudVolume(
        source, mip=mip, progress=False, use_https=True, bounded=True, fill_missing=True
    )
    labels_xyz = np.asarray(cloud_volume[:, :, :])[..., 0]
    spacing_xyz_um = np.asarray(cloud_volume.resolution, dtype=float) / 1000.0
    origin_xyz_um = np.asarray(cloud_volume.voxel_offset, dtype=float) * spacing_xyz_um
    return LabelVolume(dataset, source, mip, labels_xyz, spacing_xyz_um, origin_xyz_um)


def load_region_table(dataset: str) -> dict[int, RegionInfo]:
    source = SOURCES[dataset]
    url = f"https://storage.googleapis.com/{source.removeprefix('gs://')}/segment_properties/info"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    inline = response.json()["inline"]
    names_property = next(prop for prop in inline["properties"] if prop["type"] == "label")
    table = {}
    for raw_id, name in zip(inline["ids"], names_property["values"], strict=True):
        label_id = int(raw_id)
        _, side = split_side(name)
        table[label_id] = RegionInfo(label_id, name, side, neuropil_color(name, dataset))
    return table
