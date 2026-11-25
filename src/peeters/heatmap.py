"""The nodes as a heatmap: one row per node, as high as its share of the cells, the rows
grouped by metacluster (metaclusters and the nodes in each ordered by likeness), the
markers' medians on the right, and on the left one strip per comparison saying where each
node is more or less abundant.

``node_heatmap(res, diffs)`` draws it from a run (``run`` or ``load``) and one or more
``differential`` tables at node level; ``draw`` does the drawing from plain tables.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import style

#: the markers' medians: magma, dark (low) to bright (high)
EXPRESSION = "magma"
#: a node smaller than this share (%) is drawn at it, so every node can be seen
MIN_SHARE = 0.25
#: the colour of a node that does not differ: this much of its colour on the surface
FADE = 0.18
#: inches: the rail beside a strip, dark where the node differs
RAIL = 0.07


def marker_label(name: str) -> tuple[str, str]:
    """``cd11b#BV605`` -> (``CD11b``, ``BV605``): the marker, and its dye in small type."""
    marker, _, dye = str(name).partition("#")
    marker = marker.strip()
    if re.match(r"^cd\d", marker, flags=re.I):
        marker = "CD" + marker[2:]
    known = {"f480": "F4/80", "tim4": "Tim4", "mhcii": "MHC II", "ly6c": "Ly6C",
             "ly6g": "Ly6G", "vsig4": "Vsig4", "clec2": "Clec2", "clec4f": "Clec4f"}
    marker = known.get(marker.lower().replace(" ", ""), marker)
    dye = {"pecy7": "PE-Cy7", "pe": "PE", "apccy7": "APC-Cy7", "percpcy55": "PerCP-Cy5.5"}.get(
        dye.strip().lower().replace("-", ""), dye.strip())
    return marker, dye


def phenotypes(mc_med: pd.DataFrame, top: int = 3, z: float = 0.8) -> dict[int, str]:
    """A short name per metacluster: the markers that set it apart from the other
    metaclusters (high ⁺, low ⁻), not the ones every cell has."""
    mu, sd = mc_med.mean(), mc_med.std().replace(0, np.nan)
    out = {}
    for m, row in mc_med.iterrows():
        zs = ((row - mu) / sd).dropna()
        picks = []
        for c in zs.abs().sort_values(ascending=False).index:
            if len(picks) == top or abs(zs[c]) < z:
                break
            if zs[c] > 0 and row[c] >= 0.45:
                picks.append(f"{marker_label(c)[0]}⁺")
            elif zs[c] < 0 and row[c] <= 0.45:
                picks.append(f"{marker_label(c)[0]}⁻")
        out[m] = " ".join(picks) or "no marker stands out"
    return out


def _order(profiles: pd.DataFrame, node_mc: pd.Series) -> tuple[list[int], list[list[int]]]:
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import pdist

    def ordered(x: np.ndarray) -> list[int]:
        if len(x) < 3:
            return list(range(len(x)))
        zl = linkage(x, "average")
        return list(leaves_list(optimal_leaf_ordering(zl, pdist(x))))

    mcs = sorted(node_mc.unique())
    prof = np.vstack([profiles.loc[node_mc[node_mc == m].index].median().to_numpy()
                      for m in mcs])
    mc_order = [mcs[i] for i in ordered(prof)]
    groups = []
    for m in mc_order:
        nodes = list(node_mc[node_mc == m].index)
        groups.append([nodes[i] for i in ordered(profiles.loc[nodes].to_numpy())])
    return mc_order, groups


def _text_width(text: str, size: float) -> float:
    """How wide a line of text is, in inches, in the figures' font."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    style.apply()
    return TextPath((0, 0), text, size=size, prop=FontProperties()).get_extents().width / 72


def _spread(want: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label positions as near their places as can be, at least ``gap`` apart, within
    [lo, hi] (inches, in order)."""
    p = list(want)
    for _ in range(200):
        moved = False
        for i in range(1, len(p)):
            d = p[i] - p[i - 1]
            if d < gap:
                p[i - 1] -= (gap - d) / 2
                p[i] += (gap - d) / 2
                moved = True
        p = [min(max(v, lo), hi) for v in p]
        if not moved:
            break
    for i in range(1, len(p)):
        p[i] = max(p[i], p[i - 1] + gap)
    return p


def draw(profiles: pd.DataFrame, node_mc: pd.Series, share: pd.Series,
         comparisons: dict[str, pd.DataFrame], path: str | Path, *,
         names: dict[int, str] | None = None, title: str = "", population: str = "cells",
         alpha: float = 0.05, min_fold: float = 1.5, extra: list[str] = (),
         rule: str | None = None) -> Path:
    """``profiles``: nodes x markers (display units 0-1; the ``extra`` ones drawn after a
    gap, as not clustered on); ``node_mc``: node -> metacluster; ``share``: node -> % of
    the population; ``comparisons``: a label (``" | "`` breaks it in lines; a line every
    label has is written once over the strips) -> a node-level ``differential`` table;
    ``names``: metacluster -> the name people gave it."""
    style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.patches import Rectangle

    markers = list(profiles.columns)
    clustered = [m for m in markers if m not in extra]
    mc_order, groups = _order(profiles[clustered], node_mc)
    mc_med = pd.DataFrame({m: profiles.loc[node_mc[node_mc == m].index, clustered].median()
                           for m in mc_order}).T
    pheno = phenotypes(mc_med)
    names = names or {}
    mc_share = {m: float(share.reindex(node_mc[node_mc == m].index).fillna(0).sum())
                for m in mc_order}
    expr = plt.get_cmap(EXPRESSION)
    div = style.cmap("diverging")
    fc = {k: d.set_index("node") for k, d in comparisons.items()}
    lim = min(4.0, max([1.0] + [float(np.nanmax(np.abs(d["log2_fold"]))) for d in fc.values()]))
    dnorm = Normalize(-lim, lim)
    surface = np.array(to_rgb(style.SURFACE))

    # down the page: each node its share (at least MIN_SHARE), a gap between metaclusters
    gap = 1.0
    pos, size, span = {}, {}, {}
    t = 0.0
    for m, nodes in zip(mc_order, groups):
        start = t
        for n in nodes:
            h = max(float(share.get(n, 0.0)), MIN_SHARE)
            pos[n], size[n] = t, h
            t += h
        span[m] = (start, t)
        t += gap
    total = t - gap
    LONG, CELL = 8.6, 0.36  # inches: the height of all nodes, the width of a column
    k_in = LONG / total
    n_cmp = len(fc)
    cmp_cell, cmp_gap, ex_gap = CELL * 2.0, 0.28, 0.12
    short_cmp = n_cmp * cmp_cell
    short = short_cmp + cmp_gap + len(markers) * CELL + (ex_gap if extra else 0)

    # the left margin holds the longest metacluster label, measured in its own font
    heads, subs = [], []
    for m, nodes in zip(mc_order, groups):
        named = names.get(m)
        heads.append(f"MC{m}  {named or pheno[m]}")
        subs.append((f"{pheno[m]} · " if named else "") + f"{mc_share[m]:.1f}% · "
                    f"{len(nodes)} node{'' if len(nodes) == 1 else 's'}")
    label_in = max([_text_width(t, 8.2) for t in heads] + [_text_width(t, 6.6) for t in subs])
    left, right, top_m, bot = max(3.3, label_in + 0.75), 1.2, 1.75, 1.0
    W, H = left + short + right, top_m + LONG + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bot / H, short / W, LONG / H])
    ax.set_axis_off()
    ax.set_xlim(0, short)
    ax.set_ylim(LONG, 0)  # inches from the top

    def box(x0, x1, y0, y1, **kw):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kw))

    def mk_x(k: int) -> float:
        x = short_cmp + cmp_gap + k * CELL
        return x + (ex_gap if markers[k] in extra else 0)

    for n in pos:
        y0, y1 = pos[n] * k_in, (pos[n] + size[n]) * k_in
        for j, d in enumerate(fc.values()):
            if n not in d.index:
                continue
            r = d.loc[n]
            col = np.array(div(dnorm(np.clip(r["log2_fold"], -lim, lim)))[:3])
            differs = str(r.get("direction") or "") in ("up", "down")
            if not differs:
                col = col * FADE + surface * (1 - FADE)
            x0, x1 = j * cmp_cell, (j + 1) * cmp_cell - 0.05
            box(x0, x1 - RAIL, y0, y1, facecolor=tuple(col), edgecolor="none")
            if differs:  # the call on its own channel: a small change is pale, not faded
                box(x1 - RAIL + 0.012, x1, y0, y1, facecolor=style.INK, edgecolor="none")
        for k, mk in enumerate(markers):
            box(mk_x(k), mk_x(k) + CELL, y0, y1, facecolor=expr(float(profiles.at[n, mk])),
                edgecolor="none")
    for m, nodes in zip(mc_order, groups):
        y0, y1 = span[m][0] * k_in, span[m][1] * k_in
        for n in nodes[1:]:
            ax.plot([mk_x(0), short], [pos[n] * k_in] * 2, color=style.SURFACE, lw=0.3)
        frames = [(j * cmp_cell, (j + 1) * cmp_cell - 0.05) for j in range(n_cmp)]
        frames.append((mk_x(0), mk_x(len(clustered) - 1) + CELL))
        if extra:
            frames.append((mk_x(len(clustered)), short))
        for x0, x1 in frames:
            box(x0, x1, y0, y1, facecolor="none", edgecolor=style.INK_2, lw=0.6)

    # metacluster labels, moved apart where they would touch, a line to where they belong
    want = [(span[m][0] + span[m][1]) / 2 * k_in for m in mc_order]
    got = _spread(want, 0.34, 0.1, LONG - 0.1)
    for head, sub, w, g in zip(heads, subs, want, got):
        ax.plot([-0.05, -0.2, -0.38], [w, w, g], color=style.MUTED, lw=0.5, clip_on=False)
        ax.text(-0.45, g - 0.02, head, ha="right", va="bottom", fontsize=8.2,
                color=style.INK, clip_on=False)
        ax.text(-0.45, g + 0.02, sub, ha="right", va="top", fontsize=6.6,
                color=style.INK_2, clip_on=False)

    # headers: what all comparisons share once over the strips, what differs per strip
    split = [label.split(" | ") for label in fc]
    shared = [p for p in split[0] if all(p in s for s in split)] if len(split) > 1 else []
    for j, parts in enumerate(split):
        own = [p for p in parts if p not in shared] or parts
        ax.text(j * cmp_cell + (cmp_cell - 0.05) / 2, -0.1, "\n".join(own), ha="center",
                va="bottom", fontsize=7.0, linespacing=1.25, clip_on=False)
    if shared:
        y = -0.1 - 0.16 * (max(len(s) for s in split) - len(shared)) - 0.07
        ax.plot([0.02, short_cmp - 0.07], [y, y], color=style.MUTED, lw=0.6, clip_on=False)
        ax.text(short_cmp / 2, y - 0.05, "  ·  ".join(shared), ha="center", va="bottom",
                fontsize=7.6, clip_on=False)
    for k, mk in enumerate(markers):
        name, dye = marker_label(mk)
        c = mk_x(k) + CELL / 2
        ax.text(c, -0.1, name + ("*" if mk in extra else ""), ha="left", va="bottom",
                fontsize=8.5, rotation=60, rotation_mode="anchor", clip_on=False)
        if dye:
            ax.text(c + 0.17, -0.1, dye, ha="left", va="bottom", fontsize=5.6,
                    color=style.MUTED, rotation=60, rotation_mode="anchor", clip_on=False)

    # the share's scale, the legends, the title
    ten = 10 * k_in
    ax.plot([short + 0.25] * 2, [LONG - ten, LONG], color=style.INK, lw=1.4,
            solid_capstyle="butt", clip_on=False)
    ax.text(short + 0.35, LONG - ten / 2, f"10% of\n{population}", ha="left", va="center",
            fontsize=7, color=style.INK_2, clip_on=False)

    def bar(x_in, label, mappable):
        cax = fig.add_axes([x_in / W, 0.32 / H, 2.0 / W, 0.11 / H])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5, length=0)
        cb.set_label(label, fontsize=7, color=style.INK_2)
    bar(0.35, "log2 fold change of the node's share", plt.cm.ScalarMappable(dnorm, div))
    fig.add_artist(Rectangle((2.6 / W, 0.36 / H), 0.07 / W, 0.17 / H, facecolor=style.INK,
                             edgecolor="none", transform=fig.transFigure))
    fig.text(2.74 / W, 0.445 / H, f"differs: {rule or f'q < {alpha}, at least {min_fold}-fold'}"
             "\nwithout the bar, faded: does not differ", fontsize=6.6, color=style.INK_2,
             va="center")
    bar(4.6, "marker median on its scale (0-1)" + (";  * not clustered on" if extra else ""),
        plt.cm.ScalarMappable(Normalize(0, 1), expr))
    fig.text(0.35 / W, 1 - 0.28 / H, title, fontsize=12, color=style.INK, va="top")
    fig.text(0.35 / W, 1 - 0.56 / H,
             f"Each row is a FlowSOM node, as high as its share of the {population}; nodes "
             "grouped by metacluster, both ordered by likeness.", fontsize=7.6,
             color=style.INK_2, va="top")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return Path(path)


def comparison_label(diff: pd.DataFrame) -> str:
    """A strip's header from a ``differential`` table: its subset, the numbers of files,
    and the comparison (the last is usually the same for every strip)."""
    at = diff.attrs
    parts = [at["where_label"]] if at.get("where_label") else []
    parts.append(f"{at['n_reference']} pairs" if at.get("paired")
                 else f"{at['n_other']} vs {at['n_reference']} files")
    parts.append(f"{at['other']} vs {at['reference']}")
    return " | ".join(parts)


def node_heatmap(res, diffs, path: str | Path | None = None, *, names=None,
                 extra: bool = False, title: str = "", population: str | None = None) -> Path:
    """The node heatmap of a run with one strip per comparison: ``diffs`` a node-level
    ``differential`` table, or several (a list, or a dict label -> table). Node heights are
    the mean share over the files the comparisons compared. ``extra``: also draw the
    markers not clustered on. Written to ``<run>/node_heatmap.png`` by default."""
    if isinstance(diffs, pd.DataFrame):
        diffs = [diffs]
    if not isinstance(diffs, dict):
        diffs = {comparison_label(d): d for d in diffs}
    for d in diffs.values():
        if d.attrs.get("level") != "nodes":
            raise ValueError("the node heatmap needs node-level differential tables")
    cd = res.fsom.get_cluster_data()
    cols = list(res.markers) + (list(res.extra) if extra else [])
    profiles = pd.DataFrame(np.asarray(cd[:, cols].X, dtype=float), columns=cols,
                            index=range(1, cd.n_obs + 1))
    node_mc = res.nodes.drop_duplicates("node").set_index("node")["metacluster"]
    empty = profiles.index[profiles.isna().all(axis=1)]
    if len(empty):  # nodes no training cell reached have no medians: left out, and said
        print(f"node heatmap: {len(empty)} empty node(s) left out: "
              f"{', '.join(map(str, empty))}")
        profiles = profiles.drop(index=empty)
        node_mc = node_mc.drop(index=empty, errors="ignore")
    profiles = profiles.fillna(0.0)
    files = sorted({f for d in diffs.values()
                    for f in d.attrs["files_reference"] + d.attrs["files_other"]})
    share = res.wide("nodes", "percent").fillna(0.0).loc[files].mean()
    from .differential import rule

    first = next(iter(diffs.values())).attrs
    alpha, min_fold = first.get("alpha", 0.05), first.get("min_fold", 1.5)
    rules = {rule(d.attrs) for d in diffs.values()}
    path = Path(path) if path is not None else res.out / "node_heatmap.png"
    return draw(profiles, node_mc, share, diffs, path, names=names, title=title,
                population=population or res.params.get("population") or "cells",
                alpha=alpha, min_fold=min_fold, extra=list(res.extra) if extra else [],
                rule=" / ".join(sorted(rules)))
