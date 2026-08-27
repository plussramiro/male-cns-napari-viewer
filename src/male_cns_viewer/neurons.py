"""Download and parse selected public Male CNS SWC skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests

from .colors import neuron_color
from .config import NeuronConfig


SWC_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/"
    "skeletons-malecns/skeletons-swc/{body_id}.swc"
)
SWC_COORDINATE_UNIT_UM = 0.008


@dataclass(frozen=True)
class Neuron:
    body_id: int
    name: str
    nodes_xyz_um: np.ndarray
    parent_indices: np.ndarray
    soma_xyz_um: np.ndarray
    color: str

    @property
    def edges_zyx_um(self) -> list[np.ndarray]:
        edges = []
        for child_index, parent_index in enumerate(self.parent_indices):
            if parent_index >= 0:
                xyz = self.nodes_xyz_um[[parent_index, child_index]]
                edges.append(xyz[:, ::-1])
        return edges

    @property
    def vectors_zyx_um(self) -> np.ndarray:
        """Return napari vectors as [start, displacement] pairs."""
        edges = np.asarray(self.edges_zyx_um)
        if edges.size == 0:
            return np.empty((0, 2, 3), dtype=float)
        return np.stack((edges[:, 0], edges[:, 1] - edges[:, 0]), axis=1)


def _parse_swc(body_id: int, name: str, text: str) -> Neuron:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            fields = line.split()
            if len(fields) >= 7:
                rows.append(fields[:7])
    if not rows:
        raise ValueError(f"Skeleton {body_id} contains no SWC nodes")

    node_ids = np.asarray([int(row[0]) for row in rows])
    node_types = np.asarray([int(row[1]) for row in rows])
    xyz_um = np.asarray([[float(v) for v in row[2:5]] for row in rows]) * SWC_COORDINATE_UNIT_UM
    radii = np.asarray([float(row[5]) for row in rows])
    raw_parents = np.asarray([int(row[6]) for row in rows])
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    parents = np.asarray([index_by_id.get(parent, -1) for parent in raw_parents])

    soma_candidates = np.flatnonzero(node_types == 1)
    soma_index = int(soma_candidates[0]) if soma_candidates.size else int(np.argmax(radii))
    return Neuron(body_id, name, xyz_um, parents, xyz_um[soma_index], neuron_color(body_id))


def load_or_create_neurons(
    config: NeuronConfig,
    output_dir: Path,
    reuse_existing: bool,
    shared_cache_directories: list[Path] | None = None,
) -> list[Neuron]:
    skeleton_dir = output_dir / "skeletons"
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    neurons = []
    shared_cache_directories = shared_cache_directories or []
    for body_id in config.body_ids:
        path = skeleton_dir / f"{body_id}.swc"
        cache_candidates = [path] + [directory / "skeletons" / f"{body_id}.swc" for directory in shared_cache_directories]
        cached_path = next((candidate for candidate in cache_candidates if candidate.is_file()), None) if reuse_existing else None
        if cached_path is not None:
            text = cached_path.read_text(encoding="utf-8")
            print(f"[READ] neuron {body_id}: {cached_path}")
        else:
            print(f"[CREATE] neuron {body_id}: downloading SWC from Janelia")
            response = requests.get(SWC_URL.format(body_id=body_id), timeout=60)
            if response.status_code == 404:
                raise ValueError(f"No public Male CNS SWC found for body ID {body_id}")
            response.raise_for_status()
            text = response.text
            path.write_text(text, encoding="utf-8")
            print(f"[CREATE] wrote {path}")
        neurons.append(_parse_swc(body_id, config.names.get(str(body_id), str(body_id)), text))
    return neurons
