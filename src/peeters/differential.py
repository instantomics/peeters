"""Differential abundance between two groups of files on a FlowSOM tree: which nodes (or
metaclusters) hold more or fewer of the population's cells in one group than in the other,
and the tree with a coloured band around every node that differs.

As FlowSOM's own ``GroupStats`` (R): the unit is the file (a mouse), the value its % of the
population in each node; per node the two groups' medians, their fold change, a
Wilcoxon rank-sum test (Mann-Whitney U) and Benjamini-Hochberg over the nodes tested.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: added to both medians (percentage points) so a node empty in one group has a fold change
PSEUDO = 0.01


def _bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _files(res, by: str, where: Any) -> pd.DataFrame:
    files = res.files.set_index("file")
    if by not in files:
        raise KeyError(f"no column {by!r}; the files have {', '.join(files.columns)}")
    if where is None:
        return files
    if callable(where):
        return files[where(files)]
    keep = np.ones(len(files), dtype=bool)
    for col, value in where.items():
        values = value if isinstance(value, (list, tuple, set)) else [value]
        keep &= files[col].isin(values).to_numpy()
    return files[keep]


def rule(attrs: dict) -> str:
    """What made a unit differ, in words for a legend."""
    how = ("every pair the same way" if attrs.get("call") == "consistent"
           else f"q < {attrs.get('alpha', 0.05)}")
    return f"{how}, at least {attrs.get('min_fold', 1.5)}-fold"


def _where_label(where: Any) -> str | None:
    if where is None:
        return None
    if callable(where):
        return "a subset"
    return " · ".join("+".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v)
                      for v in where.values())


def _stem(level: str, where_label: str | None) -> str:
    """``differential_nodes``, or ``differential_nodes_Exp2`` for a subset: two subsets
    compared in one folder keep their own files."""
    if not where_label:
        return f"differential_{level}"
    slug = re.sub(r"[^A-Za-z0-9.+-]+", "_", where_label).strip("_")
    return f"differential_{level}_{slug}"


#: how a unit is called different: by its q value, or by every pair going the same way
CALLS = ("q", "consistent")


def differential(res, by: str, groups: tuple[str, str] | None = None, *,
                 level: str = "nodes", where: Any = None, paired: str | None = None,
                 call: str = "q", alpha: float = 0.05, min_fold: float = 1.5,
                 out: str | Path | None = None) -> pd.DataFrame:
    """Compare two groups of files (``by``: a column of the files' samples, e.g. genotype;
    ``groups``: (reference, other), default the two values sorted) per node or metacluster.
    ``where``: only some files, e.g. ``{"cohort": "Exp2"}`` (or a function of the files
    table).

    Unpaired (default): each group's per-file shares, Mann-Whitney, the fold change of the
    medians. ``paired``: a column that pairs a reference unit with an other one, e.g.
    ``differential(res, "part", ("WT", "EM3"), paired="source_file")`` for two parts of the
    same liver (``run(parts=...)``): per pair the log2 ratio of the two shares (each part's
    share of its own cells, so a part that fills a node as well as the other gives 0, whatever
    the mouse's chimerism), their median as the fold change, a Wilcoxon signed-rank test.

    A unit ``differs`` (``direction`` up or down; up = more in ``other``) when it changes at
    least ``min_fold``-fold and, with ``call="q"``, q < ``alpha`` (Benjamini-Hochberg over the
    units tested), or with ``call="consistent"`` (paired only) every pair goes that way: with
    few pairs no q can pass (5 pairs: p >= 0.0625), and the more units the stricter the
    correction. Written to ``<out>/differential_<level>[_<subset>].csv`` (``out`` default: the
    run's folder)."""
    from scipy.stats import mannwhitneyu, wilcoxon

    if level not in ("nodes", "metaclusters"):
        raise ValueError("level is nodes or metaclusters")
    if call not in CALLS or (call == "consistent" and not paired):
        raise ValueError("call is q, or consistent (with paired)")
    files = _files(res, by, where)
    values = [v for v in files[by].dropna().unique()]
    if groups is None:
        if len(values) != 2:
            raise ValueError(f"{by} has {len(values)} values ({values}): name the two groups")
        groups = tuple(sorted(map(str, values)))
    ref, other = (str(g) for g in groups)
    labels = files[by].astype(str)
    a, b = labels[labels == ref].index, labels[labels == other].index
    if paired:
        if paired not in files:
            raise KeyError(f"no column {paired!r} to pair by; the files have "
                           f"{', '.join(files.columns)}")
        ka, kb = files.loc[a, paired].astype(str), files.loc[b, paired].astype(str)
        keys = sorted(set(ka) & set(kb))
        if any(ka.duplicated()) or any(kb.duplicated()):
            raise ValueError(f"{paired} must name one {ref} and one {other} unit per pair")
        a = pd.Index([ka[ka == k].index[0] for k in keys])
        b = pd.Index([kb[kb == k].index[0] for k in keys])
    need = "pairs" if paired else "files each"
    if len(a) < 2 or len(b) < 2:
        raise ValueError(f"{ref}: {len(a)}, {other}: {len(b)}; a test needs at least 2 {need}")
    wide = res.wide(level, "percent").fillna(0.0)
    col = "node" if level == "nodes" else "metacluster"
    rows = []
    for unit in wide.columns:
        x, y = wide.loc[a, unit].to_numpy(), wide.loc[b, unit].to_numpy()
        tested = not (np.all(x == 0) and np.all(y == 0))
        mx, my = float(np.median(x)), float(np.median(y))
        row = {col: int(unit), f"median {ref}": mx, f"median {other}": my,
               f"mean {ref}": float(x.mean()), f"mean {other}": float(y.mean())}
        if paired:
            r = np.log2((y + PSEUDO) / (x + PSEUDO))
            lf = float(np.median(r))
            p = float(wilcoxon(r).pvalue) if tested and np.any(r != 0) else 1.0
            row.update({"log2_fold": lf, "p": p, "tested": tested, "pairs": len(r),
                        "agree": int((np.sign(r) == np.sign(lf)).sum()) if lf else 0})
        else:
            p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue) if tested else 1.0
            row.update({"log2_fold": float(np.log2((my + PSEUDO) / (mx + PSEUDO))), "p": p,
                        "tested": tested})
        rows.append(row)
    df = pd.DataFrame(rows)
    df["q"] = 1.0
    t = df["tested"].to_numpy()
    if t.any():
        df.loc[t, "q"] = _bh(df.loc[t, "p"].to_numpy())
    big = df["log2_fold"].abs() >= np.log2(min_fold)
    passed = (df["agree"] == df["pairs"]) if call == "consistent" else (df["q"] < alpha)
    df["direction"] = np.where(passed & big & df["tested"],
                               np.where(df["log2_fold"] > 0, "up", "down"), "")
    if level == "nodes":
        node_mc = res.nodes.drop_duplicates("node").set_index("node")["metacluster"]
        df.insert(1, "metacluster", df["node"].map(node_mc))
    df.attrs.update({"by": by, "reference": ref, "other": other, "level": level,
                     "alpha": alpha, "min_fold": min_fold, "n_reference": len(a),
                     "n_other": len(b), "files_reference": list(a), "files_other": list(b),
                     "paired": paired, "call": call,
                     "where": repr(where) if where is not None else None,
                     "where_label": _where_label(where)})
    target = Path(out) if out is not None else res.out
    target.mkdir(parents=True, exist_ok=True)
    df.to_csv(target / f"{_stem(level, df.attrs['where_label'])}.csv", index=False)
    n_up, n_down = int((df["direction"] == "up").sum()), int((df["direction"] == "down").sum())
    how = ("every pair the same way" if call == "consistent" else f"q < {alpha}") + \
        f", at least {min_fold}-fold"
    size = f"{len(a)} pairs" if paired else f"{len(b)} vs {len(a)} files"
    print(f"{level}: {other} vs {ref} ({size}): {n_up} up, {n_down} down ({how})")
    return df


def plot_differential(res, diff: pd.DataFrame, path: str | Path | None = None, *,
                      view: str = "MST", stars: bool = True) -> Path:
    """The tree once per group, side by side: node area = the group's median %, the stars
    the nodes' markers, and a ring around every node that differs: red = more in the other
    group, blue = fewer, deeper = a larger fold change; the numbers name the nodes (or
    metaclusters) that differ. From a metacluster-level ``diff`` every node of a
    metacluster that differs takes its ring."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from . import plots, style

    plt = plots._plt()
    at = diff.attrs
    ref, other, level, by = at["reference"], at["other"], at["level"], at["by"]
    layout, _, _ = plots.tree_layout(res.fsom, view)
    n_nodes = len(layout)
    mcs = plots._mc_of_nodes(res)
    if level == "nodes":
        per = diff.set_index("node").reindex(range(1, n_nodes + 1))
        names = {i: str(i + 1) for i in range(n_nodes)}
    else:
        per = diff.set_index("metacluster").reindex(mcs)
        per.index = range(1, n_nodes + 1)
        names = {i: f"MC{m}" for i, m in plots._mc_labels(res, layout).items()
                 for m in [mcs[i]]}
    wide = res.wide("nodes", "percent").fillna(0.0)
    sizes = {g: wide.loc[at[key]].median().reindex(range(1, n_nodes + 1)).fillna(0).to_numpy()
             for g, key in ((ref, "files_reference"), (other, "files_other"))}
    top = max(float(v.max()) for v in sizes.values()) or 1.0

    lf = per["log2_fold"].to_numpy()
    marked = (per["direction"].fillna("") != "").to_numpy()
    lim = max(1.0, float(np.nanmax(np.abs(lf[marked]))) if marked.any() else 1.0)
    cmap, norm = style.cmap("diverging"), Normalize(-lim, lim)
    band = [cmap(norm(lf[i])) if marked[i] else None for i in range(n_nodes)]
    if level == "nodes":
        text = {i: names[i] for i in np.flatnonzero(marked)}
    else:
        text = {i: t for i, t in names.items() if marked[i]}

    fig, axes, lax = plots._with_legend(plt, 2, res=res)
    for ax, g in zip(axes, (ref, other)):
        plots.draw_tree(ax, res, sizes[g], view=view, ref=top, band=band, labels=text,
                        stars=stars, labels_outside=True)
        n = at["n_reference"] if g == ref else at["n_other"]
        ax.set_title(f"{by} = {g}  ·  median of {n} " + ("units" if at.get("paired") else "files"),
                     loc="left")
    if stars:
        plots.star_legend(lax, list(res.markers))
    else:
        lax.axis("off")
    cax = fig.add_axes([0.38, 0.03, 0.24, 0.025])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label(f"ring: log2 fold change, {other} vs {ref}  ({rule(at)})", fontsize=8,
                 color=style.INK_2)
    unit = "node" if level == "nodes" else "metacluster"
    n_up, n_down = int(diff["direction"].eq("up").sum()), int(diff["direction"].eq("down").sum())
    fig.suptitle(f"{other} vs {ref}:  {n_up} {unit}{'' if n_up == 1 else 's'} more,  "
                 f"{n_down} fewer", x=0.02, ha="left", fontsize=13)
    path = Path(path) if path is not None else \
        res.out / f"{_stem(level, at.get('where_label'))}.png"
    return plots._save(fig, path)
