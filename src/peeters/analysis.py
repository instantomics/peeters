"""FlowSOM on one gated population of many FCS files, and what each file has of every
(meta)cluster: the numbers a comparison of conditions starts from.

The recipe follows FlowSOM's own protocol (Quintelier et al., Nat Protoc 2021; the R
vignette and the Python package's example): compensate, transform, train on an aggregate
with the same number of cells from every file, metacluster the nodes, then map every
file's gated events onto the trained tree and count.
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import gating as ag


@dataclass
class Input:
    path: str
    data: str | None = None  # the FCS file's data id in ALN
    name: str | None = None
    sample: str | None = None
    entry: Any = None


@dataclass
class Result:
    fsom: Any
    markers: list[str]
    extra: list[str]
    files: pd.DataFrame  # one row per file: gate counts, sample columns
    metaclusters: pd.DataFrame  # long: file x metacluster: count, percent
    nodes: pd.DataFrame  # long: file x node: count, percent
    medians: pd.DataFrame  # metacluster x marker, display units 0-1 (the trained aggregate)
    file_medians: pd.DataFrame  # long: file x metacluster x marker, data units
    out: Path
    params: dict = field(default_factory=dict)

    def wide(self, level: str = "metaclusters", value: str = "percent") -> pd.DataFrame:
        """Files as rows, (meta)clusters as columns: what a per-experiment test reads."""
        long = self.metaclusters if level == "metaclusters" else self.nodes
        col = "metacluster" if level == "metaclusters" else "node"
        return long.pivot(index="file", columns=col, values=value)


def inputs(g: ag.Gating, in_dir: str = "/in", only: list[str] | None = None,
           files: dict[str, str] | None = None) -> list[Input]:
    """The gating's FCS files as the run has them. ``files``: {data id: path} when the
    gating does not name its files or the inputs were renamed. Otherwise each file is
    ``/in/<name>``; a name that comes twice is ``<stem>-2<suffix>``, ``-3``..., in the order
    the inputs were given (as ALN names them): so give the inputs in the gating's order.
    ``only``: data ids, names or samples of the files to use (default: all)."""
    picked = [e for e in g.files
              if not only or e.data in only or e.name in only or e.sample in only]
    if not picked:
        raise ValueError(f"none of the gating's files matches {only}")
    out, seen = [], Counter()
    for e in picked:
        if files is not None:
            if e.data not in files:
                continue
            path = files[e.data]
        else:
            if not e.name:
                raise ValueError(f"the gating names no file name for {e.data}: give "
                                 "files={data id: '/in/...'}")
            seen[e.name] += 1
            n = seen[e.name]
            p = Path(e.name)
            path = str(Path(in_dir) / (e.name if n == 1 else f"{p.stem}-{n}{p.suffix}"))
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} ({e.data}) is not in the run: give it as an input")
        out.append(Input(path, e.data, e.name, e.sample, e))
    return out


def _samples_table(g: ag.Gating | None, samples: Any) -> tuple[pd.DataFrame | None, str | None]:
    match = None
    if samples is None and g is not None and g.samples:
        root = Path(g.path).parent
        while root.parent.name != "exp" and root.parent != root:
            root = root.parent
        samples, match = root / g.samples["table"], g.samples.get("match")
    if samples is None:
        return None, None
    if isinstance(samples, pd.DataFrame):
        return samples, match
    path = Path(samples)
    sep = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    return pd.read_csv(path, sep=sep, dtype=str), match


def _attach_samples(files: pd.DataFrame, table: pd.DataFrame | None, match: str | None):
    if table is None:
        return files
    keys = ["data", "name", "sample", "file"]
    cols = [match] if match else list(table.columns)
    for col in cols:
        values = set(table[col].dropna().astype(str))
        for key in keys:
            if key in files and files[key].dropna().astype(str).isin(values).all():
                merged = files.merge(table.drop_duplicates(col), left_on=key, right_on=col,
                                     how="left", suffixes=("", "_sample"))
                return merged
    print("samples: no column of the table matches the files' data ids, names or samples; "
          "not joined")
    return files


def run(gating: str | Path | None, population: str | None, markers: list[str], *,
        extra: list[str] = (), files: dict[str, str] | list[str] | None = None,
        only: list[str] | None = None, samples: Any = None, out: str | Path = "/out/flowsom",
        per_file: int = 20000, xdim: int = 10, ydim: int = 10, metaclusters: int = 12,
        seed: int = 42, by: str | None = None, in_dir: str = "/in",
        parts: dict[str, str] | None = None, plots: bool = True) -> Result:
    """Cluster ``population`` of the gating's files on ``markers`` (detectors or marker
    names) and count every file's events per node and metacluster.

    ``gating``: the experiment's gating file (``/exp/<id>/gating/x.gating.yaml``), whose
    compensation, scales and gates are used; ``None`` with ``files`` a list of paths takes
    every event of those files (already gated exports), compensated by their spillover,
    on ALN's default scales. ``extra``: markers not clustered on, of which medians per
    file and metacluster are reported (activation, reporters). ``per_file``: cells from
    each file in the training aggregate (fewer when a file has fewer). ``samples``: a
    table (path or DataFrame) joined to the files; default the gating's own ``samples``.
    ``by``: a column of it (genotype, treatment) the files are ordered and drawn by.
    ``parts``: label -> a population of the gating inside ``population`` (e.g. ``{"WT":
    "WT donor", "EM3": "EM3 donor"}`` in a competitive chimera): the tree is trained on the
    whole population, and every part of every file is counted on its own, as a row
    ``"<file> · <label>"`` with columns ``source_file`` and ``part`` (compare them with
    ``differential(res, "part", ..., paired="source_file")``).
    """
    if parts and (gating is None or population is None):
        raise ValueError("parts are populations of a gating, inside `population`")
    rng = np.random.default_rng(seed)
    g = ag.load_gating(gating) if gating is not None else None
    if g is not None:
        ins = inputs(g, in_dir, only, files if isinstance(files, dict) else None)
    else:
        if not isinstance(files, list):
            raise ValueError("without a gating give files=[paths]")
        ins = [Input(str(p), name=Path(p).name) for p in files]
        if population is not None:
            raise ValueError("a population needs a gating")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    names = list(markers) + [m for m in extra if m not in markers]
    labels: list[str] | None = None
    gated: list[np.ndarray] = []  # display units, events x names
    scales: list[list[ag.Scale]] = []
    rows = []
    in_parts: list[dict[str, np.ndarray]] = []  # per file: part -> mask over its population
    for k, inp in enumerate(ins, 1):
        ev = ag.read_fcs(inp.path)
        p = ag.Prepared(g, inp.entry, ev)
        if g is not None and population is not None:
            ms = ag.masks(p, {population, *(parts or {}).values()})
            mask = ms[population]
            counts = {f"count {n}": int(m.sum()) for n, m in ms.items()}
            in_parts.append({lab: ms[pop][mask] for lab, pop in (parts or {}).items()})
        else:
            mask = np.ones(ev.n, dtype=bool)
            counts = {}
        idx = [ev.index(m) for m in names]
        these = [ev.label(i) for i in idx]
        if labels is None:
            labels = these
        elif these != labels:
            print(f"note: {Path(inp.path).name} names the channels {these}, the first file "
                  f"{labels}; columns follow the order asked")
        x = np.column_stack([p.display(i)[mask] for i in idx]).astype(np.float32)
        gated.append(x)
        scales.append([p.scale(i) for i in idx])
        rows.append({"file": inp.name or Path(inp.path).name, "path": inp.path,
                     "data": inp.data, "sample": inp.sample, "events": ev.n,
                     "population": population, "population_events": int(mask.sum()),
                     **counts, "fcs_date": ev.keywords.get("$DATE"),
                     "cytometer": ev.keywords.get("$CYT")})
        print(f"read {k} of {len(ins)}: {rows[-1]['file']}: {int(mask.sum()):,} of "
              f"{ev.n:,} events in {population or 'the file'}", flush=True)
        p.drop()
        del ev, p
    files_df = pd.DataFrame(rows)
    dup = files_df["file"].duplicated(keep=False)  # the same file name in two runs
    files_df.loc[dup, "file"] = [f"{f} [{d or i}]" for i, (f, d) in
                                 enumerate(zip(files_df.loc[dup, "file"],
                                               files_df.loc[dup, "data"]))]
    table, match = _samples_table(g, samples)
    files_df = _attach_samples(files_df, table, match)
    base_cols = list(rows[0])
    file_names = list(files_df["file"])  # one per file, before parts make rows of their own
    # the units counted: a file, or each part of a file
    units = []  # (row of the table, file index, mask over the file's population or None)
    if parts:
        part_rows = []
        for k, row in enumerate(files_df.to_dict("records")):
            for lab in parts:
                m = in_parts[k][lab]
                unit = {**row, "file": f"{row['file']} · {lab}", "source_file": row["file"],
                        "part": lab, "population_events": int(m.sum()),
                        "file_population_events": row["population_events"]}
                part_rows.append(unit)
                units.append((unit, k, m))
        files_df = pd.DataFrame(part_rows)
        base_cols += ["source_file", "file_population_events"]
    else:
        units = [(row, k, None) for k, row in enumerate(files_df.to_dict("records"))]

    # the aggregate: the same number of cells from every file
    take = []
    for x in gated:
        n = min(per_file, x.shape[0])
        take.append(x[rng.choice(x.shape[0], n, replace=False)] if n < x.shape[0] else x)
    agg = np.concatenate(take)
    agg_file = np.repeat(np.arange(len(gated)), [t.shape[0] for t in take])
    if agg.shape[0] < xdim * ydim:
        raise ValueError(f"only {agg.shape[0]} cells to train {xdim}x{ydim} nodes")
    print(f"training FlowSOM {xdim}x{ydim}, {metaclusters} metaclusters, on {agg.shape[0]:,} "
          f"cells ({per_file:,} per file at most), {len(markers)} markers", flush=True)

    import anndata as ad
    import flowsom as fs

    adata = ad.AnnData(agg)
    adata.var_names = labels
    adata.var["marker"] = labels
    adata.obs["file"] = pd.Categorical(np.asarray(file_names)[agg_file])
    cols = labels[:len(markers)]
    fsom = fs.FlowSOM(adata, cols_to_use=cols, n_clusters=metaclusters, xdim=xdim, ydim=ydim,
                      seed=seed)
    node_mc = np.asarray(fsom.get_cluster_data().obs["metaclustering"]).astype(int)
    n_nodes = xdim * ydim

    # every file's gated events mapped onto the tree
    mc_rows, node_rows, med_rows = [], [], []
    col_idx = [labels.index(c) for c in cols]
    mapped: dict[int, np.ndarray] = {}
    for row, k, part_mask in units:
        x, sc = gated[k], scales[k]
        if k not in mapped:  # each file's cells mapped once, whatever its parts
            mapped[k] = (np.asarray(fsom.model.cluster_model.predict(x[:, col_idx])).astype(int)
                         if x.shape[0] else np.zeros(0, dtype=int))
        nodes = mapped[k]
        if part_mask is not None:
            nodes, x = nodes[part_mask], x[part_mask]
        mcs = node_mc[nodes] if len(nodes) else nodes
        total = max(len(nodes), 1)
        nc = np.bincount(nodes, minlength=n_nodes)
        mc = np.bincount(mcs, minlength=metaclusters)
        for i in range(n_nodes):
            node_rows.append({"file": row["file"], "node": i + 1,
                              "metacluster": int(node_mc[i]) + 1, "count": int(nc[i]),
                              "percent": 100.0 * nc[i] / total})
        for j in range(metaclusters):
            mc_rows.append({"file": row["file"], "metacluster": j + 1, "count": int(mc[j]),
                            "percent": 100.0 * mc[j] / total})
            sel = mcs == j
            for c, lab in enumerate(labels):
                med = float(np.median(x[sel, c])) if sel.any() else np.nan
                data = float(ag.transform(sc[c]).inverse(np.array([med]))[0]) \
                    if sel.any() else np.nan
                med_rows.append({"file": row["file"], "metacluster": j + 1, "marker": lab,
                                 "clustered": lab in cols, "median": data,
                                 "median_display": med})
    metaclusters_df = pd.DataFrame(mc_rows)
    nodes_df = pd.DataFrame(node_rows)
    file_medians = pd.DataFrame(med_rows)

    cell_mc = np.asarray(fsom.get_cell_data().obs["metaclustering"]).astype(int)
    med = pd.DataFrame(agg, columns=labels).groupby(cell_mc + 1).median()
    med.index.name = "metacluster"
    med.insert(0, "cells_in_aggregate", pd.Series(cell_mc + 1).value_counts().sort_index())
    group_cols = [c for c in files_df.columns if c not in base_cols]

    params = {"gating": str(gating) if gating else None, "population": population,
              "markers": cols, "extra": labels[len(markers):], "per_file": per_file,
              "xdim": xdim, "ydim": ydim, "metaclusters": metaclusters, "seed": seed,
              "by": by, "files": len(ins), "parts": parts or None,
              "aggregate_cells": int(agg.shape[0]),
              "scales": {lab: {"kind": s.kind, **s.p} for lab, s in zip(labels, scales[0])},
              "flowsom": getattr(fs, "__version__", None)}
    res = Result(fsom, cols, labels[len(markers):], files_df, metaclusters_df, nodes_df, med,
                 file_medians, out, params)
    files_df.to_csv(out / "files.csv", index=False)
    metaclusters_df.merge(files_df[["file", *group_cols]], on="file") \
        .to_csv(out / "metaclusters.csv", index=False)
    nodes_df.to_csv(out / "nodes.csv", index=False)
    med.to_csv(out / "metacluster_medians.csv")
    file_medians.to_csv(out / "file_medians.csv", index=False)
    (out / "params.json").write_text(json.dumps(params, indent=1, default=str))
    save_model(fsom, out / "model.pkl")
    if plots:
        from . import plots as pl

        pl.stars(res, out / "stars.png")
        pl.heatmap(res, out / "metacluster_heatmap.png")
        pl.composition(res, out / "composition.png", by=by or g and g.groups)
        if by or (g is not None and g.groups):
            pl.groups(res, by or g.groups, out / "groups.png")
    print(f"done: {out}", flush=True)
    return res


def save_model(fsom, path: str | Path) -> None:
    """The trained tree, as a pickle that loads again: a FlowSOM object pickled whole does
    not (anndata's file manager fails to unpickle), so its AnnData are kept as parts."""
    state = {k: v for k, v in vars(fsom).items() if k != "mudata"}
    state["_mods"] = {k: {"X": m.X, "obs": m.obs, "var": m.var, "uns": dict(m.uns),
                          "obsm": dict(m.obsm)} for k, m in fsom.mudata.mod.items()}
    with open(path, "wb") as fh:
        pickle.dump(state, fh)


def load_model(path: str | Path):
    """A tree ``save_model`` wrote (``model.pkl`` of a run): ``.model.cluster_model.predict``
    maps new cells, ``new_data`` and the plots work as on the original."""
    import warnings

    import anndata as ad
    import flowsom as fs
    import mudata as md

    with open(path, "rb") as fh:
        state = pickle.load(fh)
    mods = state.pop("_mods")
    fsom = object.__new__(fs.FlowSOM)
    fsom.__dict__.update(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the two modalities share var names
        fsom.mudata = md.MuData({k: ad.AnnData(**v) for k, v in mods.items()})
    return fsom


def load(folder: str | Path, out: str | Path | None = None) -> Result:
    """A run's results again from its folder (``files.csv``, ``nodes.csv``,
    ``metaclusters.csv``, ``metacluster_medians.csv``, ``file_medians.csv``, ``params.json``,
    ``model.pkl``: an earlier run's outputs given as inputs, then ``load("/in")``), to compare
    groups or draw without training again. What is written next goes to ``out`` (default the
    folder, or ``/out/flowsom`` for ``/in``)."""
    folder = Path(folder)
    out = Path(out) if out is not None else (
        Path("/out/flowsom") if folder.parts[:2] == ("/", "in") else folder)
    params = json.loads((folder / "params.json").read_text())
    fsom = load_model(folder / "model.pkl")
    files = pd.read_csv(folder / "files.csv", dtype=str)
    for col in ("events", "population_events", "file_population_events"):
        if col in files:
            files[col] = pd.to_numeric(files[col])
    mcs = pd.read_csv(folder / "metaclusters.csv")[["file", "metacluster", "count", "percent"]]
    medians = pd.read_csv(folder / "metacluster_medians.csv", index_col="metacluster")
    return Result(fsom, params["markers"], params["extra"], files, mcs,
                  pd.read_csv(folder / "nodes.csv"), medians,
                  pd.read_csv(folder / "file_medians.csv"), out, params)
