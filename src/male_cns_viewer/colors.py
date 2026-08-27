"""Generate stable colors for neuropils and selected neurons."""

from __future__ import annotations

import colorsys
import hashlib


def split_side(name: str) -> tuple[str, str]:
    for suffix, side in (("(L)", "left"), ("(R)", "right"), ("_L", "left"), ("_R", "right")):
        if name.endswith(suffix):
            return name[: -len(suffix)], side
    return name, "midline"


def _stable_hue(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def neuropil_color(name: str, dataset: str) -> str:
    base, side = split_side(name)
    hue = _stable_hue(f"{dataset}:{base}")
    lightness = {"left": 0.66, "right": 0.46}.get(side, 0.56)
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.72)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )


def neuron_color(body_id: int) -> str:
    hue = _stable_hue(f"neuron:{body_id}")
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    return "#{:02x}{:02x}{:02x}".format(
        round(red * 255), round(green * 255), round(blue * 255)
    )
