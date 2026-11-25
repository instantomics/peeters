"""The FlowSOM pictures, in one look (``style``): the tree with its metaclusters, the
markers per metacluster, every file's make-up, and the tree per group. ``draw_tree`` is
also what the differential figure draws with."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import style


def _plt():
    style.apply()
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path) -> Path:
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def tree_layout(fsom, view: str = "MST") -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """Node positions, the tree's edges and the largest node radius that does not overlap."""
    from scipy.spatial.distance import pdist

    cd = fsom.get_cluster_data()
    layout = np.asarray(cd.obsm["layout"] if view == "MST" else cd.obsm["grid"], dtype=float)
    edges = list(cd.uns["graph"].get_edgelist()) if view == "MST" else []
    r_max = float(pdist(layout).min()) / 2 * (2.0 if view == "MST" else 0.7)
    return layout, edges, r_max


def node_medians(fsom, markers: list[str]) -> np.ndarray:
    """Nodes x markers, the medians the stars draw, scaled 0-1 over all of them (FlowSOM)."""
    data = np.asarray(fsom.get_cluster_data()[:, markers].X, dtype=float)
    lo, hi = np.nanmin(data), np.nanmax(data)
    return np.nan_to_num((data - lo) / (hi - lo) if hi > lo else data * 0)


def draw_tree(ax, res, sizes: np.ndarray, *, view: str = "MST", ref: float | None = None,
              background: list[str | None] | None = None, band: list[str | None] | None = None,
              labels: dict[int, str] | None = None, stars: bool = True,
              labels_outside: bool = False) -> None:
    """One tree on ``ax``: node area = ``sizes`` (against ``ref``, default their largest),
    a star of the clustered markers in each node, an optional tint behind each node
    (``background``) and a ring around it (``band``), and text on nodes (``labels``; beside
    them, clear of the star and the ring, with ``labels_outside``)."""
    from matplotlib import collections as mc
    from matplotlib.patches import Circle, Wedge

    layout, edges, r_max = tree_layout(res.fsom, view)
    n = len(layout)
    sizes = np.nan_to_num(np.asarray(sizes, dtype=float))
    ref = ref or float(sizes.max()) or 1.0
    radii = np.maximum(np.sqrt(np.clip(sizes, 0, None) / ref) * r_max, r_max * 0.16)
    if edges:
        ax.add_collection(mc.LineCollection([[layout[a], layout[b]] for a, b in edges],
                                            colors=style.LINE, linewidths=0.7, zorder=0))
    if background is not None:
        patches = [Circle(tuple(layout[i]), r_max * 1.32) for i in range(n) if background[i]]
        cols = [c for c in background if c]
        ax.add_collection(mc.PatchCollection(patches, facecolors=cols, linewidths=0, zorder=1))
    if band is not None:
        patches = [Circle(tuple(layout[i]), radii[i] + r_max * 0.3) for i in range(n) if band[i]]
        cols = [c for c in band if c]
        ax.add_collection(mc.PatchCollection(patches, facecolors=cols, edgecolors=style.SURFACE,
                                             linewidths=0.6, zorder=2))
    ax.add_collection(mc.PatchCollection([Circle(tuple(p), r) for p, r in zip(layout, radii)],
                                         facecolors="white", edgecolors=style.MUTED,
                                         linewidths=0.5, zorder=3))
    if stars:
        markers = list(res.markers)
        heights = node_medians(res.fsom, markers) * radii[:, None]
        colors = style.marker_colors(len(markers))
        step = 360 / len(markers)
        wedges, cols = [], []
        for i in range(n):
            for k in range(len(markers)):
                if heights[i, k] > 0:
                    wedges.append(Wedge(tuple(layout[i]), heights[i, k], 90 - (k + 1) * step,
                                        90 - k * step))
                    cols.append(colors[k])
        ax.add_collection(mc.PatchCollection(wedges, facecolors=cols, edgecolors="white",
                                             linewidths=0.25, zorder=4))
    for i, text in (labels or {}).items():
        if labels_outside:  # up and to the right of the ring, out of the star's way
            d = (radii[i] + r_max * (0.3 if band is not None else 0.0)) * 0.78
            ax.text(layout[i, 0] + d, layout[i, 1] + d, text, ha="left", va="bottom",
                    fontsize=6.5, color=style.INK_2, zorder=5)
        else:
            ax.annotate(text, layout[i], ha="center", va="center", fontsize=7, color=style.INK,
                        zorder=5, bbox={"boxstyle": "round,pad=0.15", "fc": "white",
                                        "ec": "none", "alpha": 0.8})
    pad = r_max * 2
    ax.set_xlim(layout[:, 0].min() - pad, layout[:, 0].max() + pad)
    ax.set_ylim(layout[:, 1].min() - pad, layout[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def star_legend(ax, markers: list[str]) -> None:
    """One star, every wedge named at its angle: which marker each wedge is, in every tree."""
    from matplotlib.patches import Wedge

    colors = style.marker_colors(len(markers))
    step = 360 / len(markers)
    for k, (m, c) in enumerate(zip(markers, colors)):
        ax.add_patch(Wedge((0, 0), 1, 90 - (k + 1) * step, 90 - k * step, facecolor=c,
                           edgecolor="white", linewidth=0.8))
        a = np.deg2rad(90 - (k + 0.5) * step)
        ax.plot([0.75 * np.cos(a), 1.12 * np.cos(a)], [0.75 * np.sin(a), 1.12 * np.sin(a)],
                color=style.MUTED, lw=0.6)
        ax.text(1.2 * np.cos(a), 1.2 * np.sin(a), m, fontsize=8, color=style.INK,
                ha="left" if np.cos(a) > 0.15 else "right" if np.cos(a) < -0.15 else "center",
                va="center")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("markers (median per node)", fontsize=8, color=style.INK_2)


def _mc_of_nodes(res) -> np.ndarray:
    return res.nodes.drop_duplicates("node").sort_values("node")["metacluster"].to_numpy()


def _mc_labels(res, layout) -> dict[int, str]:
    """A number per metacluster, on its node nearest the metacluster's middle."""
    mcs = _mc_of_nodes(res)
    out = {}
    for m in np.unique(mcs):
        idx = np.flatnonzero(mcs == m)
        mid = layout[idx].mean(axis=0)
        out[int(idx[np.argmin(((layout[idx] - mid) ** 2).sum(1))])] = f"MC{m}"
    return out


def _with_legend(plt, n_panels: int, width: float = 7.0, res=None):
    """Panels side by side and a narrow column for the star legend; as high as the tree is
    for that width."""
    height = 6.0
    if res is not None:
        layout, _, r_max = tree_layout(res.fsom)
        span = layout.max(axis=0) - layout.min(axis=0) + 4 * r_max
        height = min(9.0, max(3.6, width * span[1] / span[0]))
    fig = plt.figure(figsize=(width * n_panels + 2.6, height + 0.9))
    gs = fig.add_gridspec(1, n_panels + 1, width_ratios=[1] * n_panels + [0.38], wspace=0.04)
    return fig, [fig.add_subplot(gs[0, i]) for i in range(n_panels)], fig.add_subplot(gs[0, -1])


def stars(res, path, view: str = "MST") -> Path:
    """The tree: node area = its share of the training cells, background = metacluster."""
    plt = _plt()
    layout, _, _ = tree_layout(res.fsom, view)
    mcs = _mc_of_nodes(res)
    cols = style.category_colors(int(mcs.max()))
    pct = np.asarray(res.fsom.get_cluster_data().obs["percentages"], dtype=float)
    fig, (ax,), lax = _with_legend(plt, 1, 8.0, res)
    draw_tree(ax, res, pct, view=view, background=[style.tint(cols[m - 1]) for m in mcs],
              labels=_mc_labels(res, layout))
    ax.set_title("FlowSOM tree: node area = share of cells, background = metacluster",
                 loc="left")
    star_legend(lax, list(res.markers))
    return _save(fig, path)


def heatmap(res, path) -> Path:
    plt = _plt()
    med = res.medians.drop(columns=["cells_in_aggregate"])
    fig, ax = plt.subplots(figsize=(0.62 * med.shape[1] + 3, 0.42 * med.shape[0] + 1.6))
    im = ax.imshow(med.to_numpy(), cmap=style.cmap("sequential"), vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(med.shape[1]), [c + ("" if c in res.markers else "*")
                                        for c in med.columns], rotation=45, ha="right")
    share = 100 * res.medians["cells_in_aggregate"] / res.medians["cells_in_aggregate"].sum()
    ax.set_yticks(range(med.shape[0]), [f"MC{i}  {s:4.1f}%" for i, s in zip(med.index, share)])
    for (r, c), v in np.ndenumerate(med.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v > 0.55 else style.INK)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("median (display units, 0-1)")
    ax.set_title("Metaclusters: median of each marker" +
                 (" (* not clustered on)" if res.extra else ""), loc="left")
    return _save(fig, path)


def composition(res, path, by: str | None = None) -> Path:
    plt = _plt()
    wide = res.wide("metaclusters", "percent")
    files = res.files.set_index("file")
    order = list(files.sort_values(by).index) if by and by in files else list(files.index)
    wide = wide.loc[order]
    cols = style.category_colors(len(wide.columns))
    fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(order) + 3.2), 4.8))
    bottom = np.zeros(len(order))
    for j, mc in enumerate(wide.columns):
        ax.bar(range(len(order)), wide[mc], bottom=bottom, color=cols[j], width=0.78,
               edgecolor=style.SURFACE, linewidth=0.8, label=f"MC{mc}")
        bottom += wide[mc].to_numpy()
    names = [f"{f}  ·  {files.loc[f, by]}" if by and by in files else f for f in order]
    ax.set_xticks(range(len(order)), names, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel(f"% of {res.params.get('population') or 'events'}")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color=style.HAIRLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, title="metacluster",
              title_fontsize=7)
    ax.set_title("Metaclusters per file", loc="left")
    return _save(fig, path)


def groups(res, by: str, path, level: str = "nodes") -> pd.DataFrame:
    """The tree once per group of ``by``, node area = the group's median % of each node (all
    on one scale), background = metacluster. To look at; the test is ``differential``."""
    plt = _plt()
    files = res.files.set_index("file")
    if by not in files:
        raise KeyError(f"no column {by!r}; the files have {', '.join(files.columns)}")
    wide = res.wide("nodes", "percent").fillna(0.0)
    med = wide.groupby(files.loc[wide.index, by].astype(str)).median()
    layout, _, _ = tree_layout(res.fsom)
    mcs = _mc_of_nodes(res)
    cols = style.category_colors(int(mcs.max()))
    fig, axes, lax = _with_legend(plt, len(med), res=res)
    ref = float(med.to_numpy().max())
    for ax, (name, row) in zip(axes, med.iterrows()):
        draw_tree(ax, res, row.to_numpy(), ref=ref, labels=_mc_labels(res, layout),
                  background=[style.tint(cols[m - 1]) for m in mcs])
        ax.set_title(f"{by} = {name}  ·  median of {int((files[by].astype(str) == name).sum())}"
                     " files", loc="left")
    star_legend(lax, list(res.markers))
    _save(fig, path)
    return med
