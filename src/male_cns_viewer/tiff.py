"""Export and reload Male CNS neuropil label TIFFs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tifffile

from .data import LabelVolume, RegionInfo


def artifact_paths(dataset: str, mip: int, output_dir: Path) -> tuple[Path, Path]:
    stem = output_dir / f"male_cns_{dataset}_rois_mip{mip}"
    return stem.with_suffix(".tif"), stem.with_suffix(".json")


def load_label_tiff(dataset: str, mip: int, output_dir: Path):
    tiff_path, json_path = artifact_paths(dataset, mip, output_dir)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata["dataset"] != dataset or metadata["mip"] != mip:
        raise ValueError("Cached metadata does not match the requested dataset/MIP")
    labels_zyx = tifffile.imread(tiff_path)
    if labels_zyx.shape != tuple(metadata["shape_zyx"]):
        raise ValueError("Cached TIFF shape does not match its metadata")
    spacing_xyz_um = np.asarray(metadata["scale_zyx_um"], dtype=float)[::-1]
    origin_xyz_um = np.asarray(metadata["origin_zyx_um"], dtype=float)[::-1]
    volume = LabelVolume(
        dataset, metadata["source"], mip,
        np.transpose(labels_zyx, (2, 1, 0)), spacing_xyz_um, origin_xyz_um
    )
    table = {item["label_id"]: RegionInfo(**item) for item in metadata["regions"]}
    return volume, table


def export_label_tiff(volume: LabelVolume, table: dict[int, RegionInfo], output_dir: Path):
    labels_zyx = np.transpose(volume.labels_xyz, (2, 1, 0)).astype(np.uint16)
    output_dir.mkdir(parents=True, exist_ok=True)
    tiff_path, json_path = artifact_paths(volume.dataset, volume.mip, output_dir)
    tifffile.imwrite(
        tiff_path, labels_zyx, imagej=True, compression="zlib",
        resolution=(1 / volume.spacing_xyz_um[0], 1 / volume.spacing_xyz_um[1]),
        metadata={"axes": "ZYX", "unit": "um", "spacing": float(volume.spacing_xyz_um[2])},
    )
    metadata = {
        "dataset": volume.dataset, "source": volume.source, "mip": volume.mip,
        "axes": "ZYX", "shape_zyx": list(labels_zyx.shape),
        "scale_zyx_um": volume.spacing_xyz_um[::-1].tolist(),
        "origin_zyx_um": volume.origin_xyz_um[::-1].tolist(),
        "background_label": 0,
        "regions": [asdict(table[key]) for key in sorted(table)],
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tiff_path, json_path
