"""Run the Male CNS workflow with TensorStore-backed N5 streaming."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

# Allow `python main.py` from a fresh clone; an editable install is still recommended.
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from male_cns_viewer.config import load_config
from male_cns_viewer.data import load_label_volume, load_region_table
from male_cns_viewer.em_tensorstore import open_em_pyramid
from male_cns_viewer.napari_viewer import show_in_napari
from male_cns_viewer.neurons import load_or_create_neurons
from male_cns_viewer.tiff import artifact_paths, export_label_tiff, load_label_tiff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the optimized multimodal Male CNS viewer."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    return parser


def main() -> None:
    started = perf_counter()
    config = load_config(build_parser().parse_args().config)

    def timing(label: str, since: float) -> float:
        now = perf_counter()
        if config.performance.show_timings:
            print(f"[TIMING] {label}: {now - since:.2f} s")
        return now

    # 1. Load or create the global brain/VNC neuropil label volumes.
    datasets = {}
    stage = perf_counter()
    for region in config.dataset.regions:
        cache_directories = [config.output.directory] + config.output.shared_cache_directories
        if config.output.reuse_existing:
            cached_directory = next(
                (
                    directory
                    for directory in cache_directories
                    if all(path.is_file() for path in artifact_paths(region, config.dataset.mip, directory))
                ),
                None,
            )
        else:
            cached_directory = None
        if cached_directory is not None:
            try:
                datasets[region] = load_label_tiff(
                    region, config.dataset.mip, cached_directory
                )
                print(f"[READ] {region} MIP {config.dataset.mip}: {cached_directory}")
                continue
            except (KeyError, OSError, TypeError, ValueError) as error:
                print(f"[CREATE] {region}: cached files are invalid ({error})")

        print(f"[CREATE] {region} MIP {config.dataset.mip}: downloading from Janelia")
        volume = load_label_volume(region, mip=config.dataset.mip)
        region_table = load_region_table(region)
        datasets[region] = (volume, region_table)
        created = export_label_tiff(volume, region_table, config.output.directory)
        for path in created:
            print(f"[CREATE] wrote {path}")
    stage = timing("neuropil layers", stage)

    # 2. Load only the selected public SWC skeletons and cache them locally.
    neurons = load_or_create_neurons(
        config.neurons,
        config.output.directory,
        config.output.reuse_existing,
        config.output.shared_cache_directories,
    )
    stage = timing("selected neurons", stage)

    # 3. Open lazy remote arrays. No whole-EM TIFF or NumPy volume is created.
    em_stream = open_em_pyramid(config.em, config.performance)
    stage = timing("remote EM metadata and lazy pyramids", stage)

    # 4. Combine image, labels, skeletons, and somas in the same napari scene.
    if config.napari.launch:
        timing("workflow before napari", started)
        show_in_napari(datasets, neurons, em_stream, config.neurons, config.napari)


if __name__ == "__main__":
    main()
