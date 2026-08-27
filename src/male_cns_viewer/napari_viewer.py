"""Combine EM, neuropils, neuron skeletons, and somas in napari."""

from __future__ import annotations

import numpy as np

from .config import NapariConfig, NeuronConfig
from .data import LabelVolume, RegionInfo
from .em_tensorstore import EmStream
from .neurons import Neuron


def _features(table: dict[int, RegionInfo]) -> dict[str, list[object]]:
    max_id = max(table, default=0)
    return {
        "label_id": list(range(max_id + 1)),
        "name": ["background"] + [table[i].name if i in table else "unused" for i in range(1, max_id + 1)],
        "side": ["none"] + [table[i].side if i in table else "none" for i in range(1, max_id + 1)],
    }


def show_in_napari(
    datasets: dict[str, tuple[LabelVolume, dict[int, RegionInfo]]],
    neurons: list[Neuron],
    em_stream: EmStream | None,
    neuron_config: NeuronConfig,
    config: NapariConfig,
) -> None:
    import napari

    create_napari_viewer(datasets, neurons, em_stream, neuron_config, config)
    napari.run()


def create_napari_viewer(
    datasets, neurons, em_stream, neuron_config: NeuronConfig, config: NapariConfig
):
    import napari

    viewer = napari.Viewer(ndisplay=config.ndisplay, title="Male CNS: neuropils + neurons + EM")

    # Original, non-CLAHE N5 data. TensorStore reads only requested chunks.
    if em_stream is not None:
        overview_mip = em_stream.overview_3d_mip
        viewer.add_image(
            em_stream.overview_3d_zyx,
            multiscale=False,
            name=f"Original EM N5 — 3D overview s{overview_mip}",
            scale=tuple(em_stream.overview_3d_spacing_zyx_um),
            axis_labels=("Z", "Y", "X"), units=("um", "um", "um"),
            opacity=config.em_opacity, blending="translucent",
            contrast_limits=em_stream.contrast_limits,
            rendering="mip", visible=em_stream.visible_on_start,
            metadata={"source": em_stream.source, "mip": overview_mip},
        )
        viewer.add_image(
            em_stream.detail_2d_pyramid_zyx, multiscale=True,
            name="Original EM N5 — 2D full-resolution detail",
            scale=tuple(em_stream.detail_2d_finest_spacing_zyx_um),
            translate=tuple(em_stream.origin_zyx_um),
            axis_labels=("Z", "Y", "X"), units=("um", "um", "um"),
            opacity=1.0, blending="translucent",
            contrast_limits=em_stream.contrast_limits,
            rendering="mip", visible=False,
            metadata={"source": em_stream.source, "mips": em_stream.detail_2d_mip_levels},
        )
        if em_stream.fast_2d_pyramid_zyx is not None:
            viewer.add_image(
                em_stream.fast_2d_pyramid_zyx,
                multiscale=True,
                name="Fast EM CLAHE-JPEG — 2D navigation",
                scale=tuple(em_stream.fast_2d_finest_spacing_zyx_um),
                translate=tuple(em_stream.origin_zyx_um),
                axis_labels=("Z", "Y", "X"), units=("um", "um", "um"),
                opacity=1.0, blending="translucent", visible=False,
                contrast_limits=(0, 255),
                metadata={"mips": em_stream.fast_2d_mip_levels},
            )
        print(f"[NAPARI] 3D EM overview uses N5 s{overview_mip}")
        print("[NAPARI] enable the 2D detail pyramid after switching the viewer to 2D")
        if em_stream.fast_2d_pyramid_zyx is not None:
            print("[NAPARI] use Fast EM CLAHE-JPEG for responsive 2D navigation")

    # The categorical brain/VNC volumes provide global neuropil context.
    for dataset, (volume, table) in datasets.items():
        labels_zyx = np.transpose(volume.labels_xyz, (2, 1, 0)).astype(np.uint16)
        colors = {0: "#00000000", None: "#808080"}
        colors.update({key: info.color for key, info in table.items()})
        viewer.add_labels(
            labels_zyx, name=f"Male CNS — {dataset.upper()} neuropils",
            scale=tuple(volume.spacing_xyz_um[::-1]),
            translate=tuple(volume.origin_xyz_um[::-1]),
            axis_labels=("Z", "Y", "X"), units=("um", "um", "um"),
            opacity=config.neuropil_opacity, blending="translucent",
            rendering="iso_categorical", features=_features(table), colormap=colors,
            metadata={"source": volume.source, "mip": volume.mip},
        )

    # All SWC parent-child segments share one efficient Vectors layer per neuron.
    for neuron in neurons:
        if neuron_config.show_skeletons and neuron.vectors_zyx_um.size:
            viewer.add_vectors(
                neuron.vectors_zyx_um, name=f"Skeleton — {neuron.name}",
                edge_color=neuron.color, edge_width=0.35, length=1.0,
                units=("um", "um", "um"),
                metadata={"body_id": neuron.body_id, "name": neuron.name},
            )
        if neuron_config.show_somas:
            viewer.add_points(
                neuron.soma_xyz_um[::-1][None, :], name=f"Soma — {neuron.name}",
                size=neuron_config.soma_size_um, face_color=neuron.color,
                units=("um", "um", "um"),
                border_color="white", features={"body_id": [neuron.body_id], "name": [neuron.name]},
            )

    if neurons:
        viewer.camera.center = tuple(neurons[0].soma_xyz_um[::-1])

    if config.show_features_table:
        try:
            viewer.window.add_plugin_dock_widget("napari", "Features table widget")
        except LookupError:
            print("[INFO] napari Features table widget is unavailable")

    # Napari's play button advances by the Dims step. Multiplying the native
    # Z step lets remote EM playback skip sections without changing the data,
    # the available MIPs, or the resolution used to render each requested cut.
    z_axis = 0  # Every volume is presented to napari in ZYX order.
    z_min, z_max, native_z_step_um = viewer.dims.range[z_axis]
    playback_z_step_um = native_z_step_um * config.playback_step_slices
    viewer.dims.set_range(z_axis, (z_min, z_max, playback_z_step_um))
    print(
        "[NAPARI] Z playback step: "
        f"{config.playback_step_slices} native slice(s) = "
        f"{playback_z_step_um:g} um per frame"
    )
    return viewer
