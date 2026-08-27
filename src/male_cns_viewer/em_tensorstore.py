"""Stream the original uncompressed Male CNS N5 pyramid with TensorStore."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import dask
import dask.array as da
import numpy as np
import tensorstore as ts

from .config import EmConfig, PerformanceConfig


class TensorStoreArray:
    """Expose asynchronous TensorStore reads through Dask's array protocol."""

    def __init__(self, store: ts.TensorStore):
        self.store = store
        self.shape = tuple(int(value) for value in store.shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(store.dtype.numpy_dtype)

    def __getitem__(self, index):
        # Only the slice requested by Dask is transferred from Janelia.
        return self.store[index].read().result()


class TensorStoreChannelArray(TensorStoreArray):
    """Expose the first channel of a rank-4 Neuroglancer image as XYZ."""

    def __init__(self, store: ts.TensorStore):
        super().__init__(store)
        self.shape = self.shape[:3]
        self.ndim = 3

    def __getitem__(self, index):
        return self.store[index + (0,)].read().result()


@dataclass(frozen=True)
class EmStream:
    detail_2d_pyramid_zyx: list[da.Array]
    detail_2d_mip_levels: list[int]
    detail_2d_finest_spacing_zyx_um: np.ndarray
    overview_3d_zyx: da.Array
    overview_3d_mip: int
    overview_3d_spacing_zyx_um: np.ndarray
    fast_2d_pyramid_zyx: list[da.Array] | None
    fast_2d_mip_levels: list[int]
    fast_2d_finest_spacing_zyx_um: np.ndarray | None
    origin_zyx_um: np.ndarray
    source: str
    visible_on_start: bool
    contrast_limits: tuple[float, float]


def _chunk_shape(store: ts.TensorStore) -> tuple[int, int, int]:
    read_chunk = store.chunk_layout.read_chunk.shape
    return tuple(
        min(int(chunk), int(size))
        for chunk, size in zip(read_chunk, store.shape, strict=True)
    )


def _open_level(source: str, mip: int, context: ts.Context):
    store = ts.open(
        {
            "driver": "n5",
            "kvstore": f"{source.rstrip('/')}/s{mip}/",
            "recheck_cached_metadata": "open",
        },
        open=True,
        context=context,
    ).result()
    remote = TensorStoreArray(store)
    lazy_xyz = da.from_array(
        remote,
        chunks=_chunk_shape(store),
        asarray=False,
        fancy=False,
        name=f"male-cns-original-n5-s{mip}",
    )
    return lazy_xyz.transpose(2, 1, 0), store


def _open_fast_level(source: str, mip: int, context: ts.Context):
    store = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": f"{source.rstrip('/')}/",
            "scale_index": mip,
            "recheck_cached_metadata": "open",
        },
        open=True,
        context=context,
    ).result()
    remote = TensorStoreChannelArray(store)
    chunks_xyz = tuple(
        min(int(chunk), int(size))
        for chunk, size in zip(store.chunk_layout.read_chunk.shape[:3], remote.shape, strict=True)
    )
    lazy_xyz = da.from_array(
        remote,
        chunks=chunks_xyz,
        asarray=False,
        fancy=False,
        name=f"male-cns-fast-clahe-mip{mip}",
    )
    return lazy_xyz.transpose(2, 1, 0), store


def open_em_pyramid(config: EmConfig, performance: PerformanceConfig) -> EmStream | None:
    if not config.enabled:
        print("[SKIP] original N5 EM streaming disabled")
        return None

    dask.config.set(scheduler="threads", num_workers=performance.dask_workers)
    context = ts.Context({
        "cache_pool": {"total_bytes_limit": int(config.cache_gb * 1024**3)},
        "data_copy_concurrency": {"limit": performance.n5_open_workers},
    })
    opened = {}
    requested_mips = sorted(set(config.detail_2d_mips + [config.overview_3d_mip]))
    workers = min(performance.n5_open_workers, len(requested_mips))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_open_level, config.source, mip, context): mip
            for mip in requested_mips
        }
        for future in as_completed(futures):
            mip = futures[future]
            lazy_zyx, store = future.result()
            opened[mip] = lazy_zyx
            spacing_um = 0.008 * 2**mip
            print(
                f"[TENSORSTORE] N5 s{mip}: {tuple(lazy_zyx.shape)} "
                f"at {spacing_um:g} um/voxel; chunks={tuple(reversed(_chunk_shape(store)))}"
            )

    fast_opened = {}
    if config.fast_2d_enabled:
        fast_workers = min(performance.n5_open_workers, len(config.fast_2d_mips))
        with ThreadPoolExecutor(max_workers=fast_workers) as executor:
            futures = {
                executor.submit(_open_fast_level, config.fast_2d_source, mip, context): mip
                for mip in config.fast_2d_mips
            }
            for future in as_completed(futures):
                mip = futures[future]
                lazy_zyx, store = future.result()
                fast_opened[mip] = lazy_zyx
                print(f"[FAST] CLAHE-JPEG MIP {mip}: {tuple(lazy_zyx.shape)}")

    detail_mips = config.detail_2d_mips
    finest_mip = detail_mips[0]
    finest_spacing_um = 0.008 * 2**finest_mip
    print("[TENSORSTORE] lazy remote pyramid ready; no whole EM array was downloaded")
    return EmStream(
        detail_2d_pyramid_zyx=[opened[mip] for mip in detail_mips],
        detail_2d_mip_levels=list(detail_mips),
        detail_2d_finest_spacing_zyx_um=np.full(3, finest_spacing_um),
        overview_3d_zyx=opened[config.overview_3d_mip],
        overview_3d_mip=config.overview_3d_mip,
        overview_3d_spacing_zyx_um=np.full(3, 0.008 * 2**config.overview_3d_mip),
        fast_2d_pyramid_zyx=(
            [fast_opened[mip] for mip in config.fast_2d_mips]
            if config.fast_2d_enabled else None
        ),
        fast_2d_mip_levels=list(config.fast_2d_mips) if config.fast_2d_enabled else [],
        fast_2d_finest_spacing_zyx_um=(
            np.full(3, 0.008 * 2**config.fast_2d_mips[0])
            if config.fast_2d_enabled else None
        ),
        origin_zyx_um=np.zeros(3),
        source=config.source,
        visible_on_start=config.visible_on_start,
        contrast_limits=tuple(config.contrast_limits),
    )
