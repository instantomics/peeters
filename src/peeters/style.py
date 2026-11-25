"""One look for every figure: a validated categorical order, a blue <-> red diverging pair
with a grey middle, a one-hue blue ramp, quiet ink and lines.

Markers keep one colour each in every figure. They never take the diverging pair's blue
or red (those are the differential bands), and their identity never rests on colour
alone: the wedge's angle in a star is the same marker everywhere, and the star legend
names every angle.
"""

from __future__ import annotations

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
LINE = "#c3c2b7"
HAIRLINE = "#e1e0d9"

#: the categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
               "#6250d6", "#e34948"]
#: markers: the categorical order without the diverging pair's blue and red
MARKER_HUES = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
DIVERGING = ["#1c5cab", "#2a78d6", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#a92b2b"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    a = np.array([int(hex_a[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(hex_b[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    c = np.rint(a * (1 - t) + b * t).astype(int)
    return "#" + "".join(f"{v:02x}" for v in c)


def marker_colors(n: int) -> list[str]:
    """n markers: the six hues, then the same hues darker, then lighter (their angle in the
    star and the labelled legend tell same-hue markers apart)."""
    steps = MARKER_HUES + [_mix(h, INK, 0.38) for h in MARKER_HUES] + \
        [_mix(h, "#ffffff", 0.45) for h in MARKER_HUES]
    return [steps[i % len(steps)] for i in range(n)]


def tint(hex_color: str, t: float = 0.72) -> str:
    """A light tint of a colour on the surface (backgrounds behind the nodes)."""
    return _mix(hex_color, SURFACE, t)


def category_colors(n: int) -> list[str]:
    """n categories (metaclusters): the categorical order, then its darker steps; past
    eight, the categories are also labelled by number wherever they are drawn."""
    steps = CATEGORICAL + [_mix(h, INK, 0.35) for h in CATEGORICAL]
    return [steps[i % len(steps)] for i in range(n)]


def cmap(name: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        f"peeters_{name}", DIVERGING if name == "diverging" else SEQUENTIAL)


def apply() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "regular",
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": LINE,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
