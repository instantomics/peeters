"""Synthetic FCS files with known populations: the gating's counts are exact, FlowSOM finds
the three cell types, and the KO files' shift between two of them is in the numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peeters import gating as ag
from peeters import inputs, run

CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "V450-A", "APC-A", "PE-A", "FITC-A"]
MARKERS = ["", "", "", "CD45", "F4/80", "Tim4", "Clec2"]
SPILL = "4,V450-A,APC-A,PE-A,FITC-A,1,0,0,0,0,1,0.02,0,0,0,1,0.05,0,0,0.1,1"


def write_fcs(path, events, channels, markers, meta, top=262144.0):
    events = np.asarray(events, dtype="<f4")
    n_tot, n_par = events.shape
    kw = {"$BYTEORD": "1,2,3,4", "$DATATYPE": "F", "$MODE": "L", "$NEXTDATA": "0",
          "$PAR": str(n_par), "$TOT": str(n_tot), "$FIL": Path(path).name, **meta}
    for i, (c, m) in enumerate(zip(channels, markers), start=1):
        kw[f"$P{i}N"], kw[f"$P{i}B"], kw[f"$P{i}E"], kw[f"$P{i}R"] = c, "32", "0,0", str(int(top))
        if m:
            kw[f"$P{i}S"] = m
    data = events.tobytes()

    def text_for(begin, end):
        full = {**kw, "$BEGINDATA": f"{begin:>12}", "$ENDDATA": f"{end:>12}"}
        return ("/" + "".join(f"{k.replace('/', '//')}/{str(v).replace('/', '//')}/"
                              for k, v in full.items())).encode()

    start = 58 + len(text_for(0, 0))
    text = text_for(start, start + len(data) - 1)
    header = b"FCS3.1    " + b"".join(f"{v:>8}".encode() for v in (
        58, 58 + len(text) - 1, start, start + len(data) - 1, 0, 0))
    Path(path).write_bytes(header + text + data)


#: true marker values (compensated): CD45, F4/80, Tim4, Clec2
TYPES = {"KC": (8000, 6000, 9000, 300), "moKC": (8000, 6000, 150, 7000),
         "mono": (8000, 100, 120, 200)}


def make_file(path, rng, shares, n=6000):
    """Events: 10% debris (small), 10% doublets (FSC-H low), 10% CD45-, the rest the cell
    types in ``shares``. Returns the true counts."""
    n_debris, n_dbl, n_neg = n // 10, n // 10, n // 10
    n_cells = n - n_debris - n_dbl - n_neg
    kinds = rng.choice(list(TYPES), n_cells, p=[shares[k] for k in TYPES])
    rows, truth = [], {k: int((kinds == k).sum()) for k in TYPES}

    def ev(fsc, ssc, single, cd45, f480, tim4, clec2):
        fsch = fsc * (0.8 if single else 0.35)
        return [fsc, fsch, ssc, cd45, f480, tim4, clec2]

    for _ in range(n_debris):
        rows.append(ev(rng.uniform(2000, 20000), rng.uniform(1000, 20000), True, 50, 50, 50, 50))
    for _ in range(n_dbl):
        rows.append(ev(rng.uniform(80000, 200000), rng.uniform(40000, 120000), False,
                       8000, 6000, 9000, 300))
    for _ in range(n_neg):
        rows.append(ev(rng.uniform(80000, 200000), rng.uniform(40000, 120000), True,
                       30, 80, 60, 40))
    for k in kinds:
        cd45, f480, tim4, clec2 = (v * rng.lognormal(0, 0.15) for v in TYPES[k])
        rows.append(ev(rng.uniform(80000, 200000), rng.uniform(40000, 120000), True,
                       cd45, f480, tim4, clec2))
    true = np.array(rows, dtype=np.float64)
    spill = np.array([float(v) for v in SPILL.split(",")[5:]]).reshape(4, 4)
    raw = true.copy()
    raw[:, 3:7] = true[:, 3:7] @ spill  # what the cytometer measures
    rows = rng.permutation(raw)
    write_fcs(path, rows, CHANNELS, MARKERS, {"$SPILLOVER": SPILL, "$CYT": "test"})
    return {"Cells": n - n_debris, "Singlets": n - n_debris - n_dbl,
            "CD45+": n_cells, **truth}


GATING = """format: aln-gating/1
title: test
files:
{files}
compensation: file
samples: {{table: samples.csv, match: data}}
groups: genotype
populations:
  Cells:
    parent: null
    rectangle: {{x: FSC-A, y: SSC-A, x_min: 50000, y_min: 25000}}
  Singlets:
    parent: Cells
    polygon:
      x: FSC-A
      y: FSC-H
      points: [[40000, 20000], [262000, 140000], [262000, 262000], [40000, 60000]]
  CD45+:
    parent: Singlets
    range: {{x: CD45, min: 1000}}
"""


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp") / "exp" / "exp_test000000"
    (root / "gating").mkdir(parents=True)
    ind = tmp_path_factory.mktemp("in")
    rng = np.random.default_rng(1)
    truths, entries, samples = {}, [], []
    for i in range(6):
        ko = i >= 3
        shares = {"KC": 0.2, "moKC": 0.6, "mono": 0.2} if ko else \
            {"KC": 0.6, "moKC": 0.2, "mono": 0.2}
        # two runs that both called their tube Tube_001: ALN names the second one -2
        name = "Tube_001.fcs" if i in (0, 3) else f"Tube_{i + 1:03d}.fcs"
        target = ind / name if not (ind / name).exists() else ind / "Tube_001-2.fcs"
        truths[f"dat_test{i:06d}"] = make_file(target, rng, shares)
        entries.append(f"- {{data: dat_test{i:06d}, name: {name}}}")
        samples.append({"data": f"dat_test{i:06d}", "genotype": "KO" if ko else "WT"})
    (root / "gating" / "t.gating.yaml").write_text(GATING.format(files="\n".join(entries)))
    pd.DataFrame(samples).to_csv(root / "samples.csv", index=False)
    return root, ind, truths


def test_inputs_follow_alns_names(experiment):
    root, ind, _ = experiment
    g = ag.load_gating(root / "gating" / "t.gating.yaml")
    got = [Path(i.path).name for i in inputs(g, str(ind))]
    assert got == ["Tube_001.fcs", "Tube_002.fcs", "Tube_003.fcs", "Tube_001-2.fcs",
                   "Tube_005.fcs", "Tube_006.fcs"]


def test_gate_counts_are_exact(experiment):
    root, ind, truths = experiment
    g = ag.load_gating(root / "gating" / "t.gating.yaml")
    for inp in inputs(g, str(ind)):
        p = ag.Prepared(g, inp.entry, ag.read_fcs(inp.path))
        ms = ag.masks(p)
        t = truths[inp.data]
        assert {n: int(m.sum()) for n, m in ms.items()} == \
            {k: t[k] for k in ("Cells", "Singlets", "CD45+")}


def test_flowsom_finds_types_and_the_shift(experiment, tmp_path):
    root, ind, truths = experiment
    res = run(root / "gating" / "t.gating.yaml", "CD45+", ["F4/80", "Tim4", "Clec2"],
              extra=["CD45"], in_dir=str(ind), out=tmp_path, per_file=1500, xdim=5, ydim=5,
              metaclusters=3, seed=1)
    assert list(res.files["population_events"]) == [t["CD45+"] for t in truths.values()]
    assert set(res.files["genotype"]) == {"WT", "KO"}
    med = res.medians
    kc = med["Tim4"].idxmax()
    mokc = med["Clec2"].idxmax()
    mono = med["F4/80"].idxmin()
    assert len({kc, mokc, mono}) == 3
    wide = res.wide()
    for f, d in zip(res.files["file"], res.files["data"]):
        t = truths[d]
        assert wide.loc[f, kc] == pytest.approx(100 * t["KC"] / t["CD45+"], abs=1.0)
        assert wide.loc[f, mokc] == pytest.approx(100 * t["moKC"] / t["CD45+"], abs=1.0)
    by = res.files.set_index("file")["genotype"]
    assert wide.loc[by == "KO", kc].mean() < 30 < wide.loc[by == "WT", kc].mean()
    for name in ("files.csv", "metaclusters.csv", "nodes.csv", "metacluster_medians.csv",
                 "file_medians.csv", "params.json", "model.pkl", "stars.png",
                 "metacluster_heatmap.png", "composition.png", "groups.png"):
        assert (tmp_path / name).exists(), name
    fm = res.file_medians
    cd45 = fm[(fm.marker == "CD45") & (fm.metacluster == kc)]["median"]
    assert cd45.between(6000, 10000).all()  # data units, compensated


def test_differential_finds_the_shift_and_draws_the_bands(tmp_path):
    """Five WT and five KO files: the KC nodes are down in KO, the moKC nodes up, the
    monocytes (the same share in both) unchanged; the tree with the bands is drawn."""
    from peeters import differential, load, plot_differential

    root = tmp_path / "exp" / "exp_diff000000"
    (root / "gating").mkdir(parents=True)
    ind = tmp_path / "in"
    ind.mkdir()
    rng = np.random.default_rng(7)
    entries, samples = [], []
    for i in range(10):
        ko = i % 2 == 1
        shares = {"KC": 0.2, "moKC": 0.6, "mono": 0.2} if ko else \
            {"KC": 0.6, "moKC": 0.2, "mono": 0.2}
        make_file(ind / f"T{i}.fcs", rng, shares, n=3000)
        entries.append(f"- {{data: dat_diff{i:06d}, name: T{i}.fcs}}")
        samples.append({"data": f"dat_diff{i:06d}", "genotype": "KO" if ko else "WT",
                        "cohort": "A" if i < 6 else "B"})
    (root / "gating" / "d.gating.yaml").write_text(GATING.format(files="\n".join(entries)))
    pd.DataFrame(samples).to_csv(root / "samples.csv", index=False)
    res = run(root / "gating" / "d.gating.yaml", "CD45+", ["F4/80", "Tim4", "Clec2"],
              in_dir=str(ind), out=tmp_path / "out", per_file=800, xdim=6, ydim=6,
              metaclusters=3, seed=3, plots=False)

    mc = differential(res, "genotype", ("WT", "KO"), level="metaclusters")
    kc, mokc, mono = (res.medians[m].idxmax() if m != "F4/80" else res.medians[m].idxmin()
                      for m in ("Tim4", "Clec2", "F4/80"))
    d = mc.set_index("metacluster")["direction"]
    assert (d[kc], d[mokc], d[mono]) == ("down", "up", "")
    assert mc.attrs["n_reference"] == mc.attrs["n_other"] == 5
    assert (tmp_path / "out" / "differential_metaclusters.csv").exists()

    nodes = differential(res, "genotype", ("WT", "KO"))
    up = nodes[nodes.direction == "up"]
    assert len(up) and set(up["metacluster"]) == {mokc}
    assert set(nodes[nodes.direction == "down"]["metacluster"]) == {kc}
    for diff in (nodes, mc):
        assert plot_differential(res, diff).exists()

    # a subset of the files: cohort A has 3 WT and 3 KO, too few for q < 0.05 at 3 vs 3
    sub = differential(res, "genotype", ("WT", "KO"), level="metaclusters",
                       where={"cohort": "A"}, out=tmp_path / "sub")
    assert sub.attrs["n_reference"] == 3 and (sub["direction"] == "").all()

    # the node heatmap: one strip per comparison, labelled by subset, files and groups
    from peeters import node_heatmap
    from peeters.heatmap import comparison_label

    a = differential(res, "genotype", ("WT", "KO"), where={"cohort": "A"}, out=tmp_path / "sub")
    b = differential(res, "genotype", ("WT", "KO"), where={"cohort": "B"}, out=tmp_path / "sub")
    assert comparison_label(a) == "A | 3 vs 3 files | KO vs WT"
    assert node_heatmap(res, [nodes, a, b], names={kc: "KC-like"}).exists()
    with pytest.raises(ValueError, match="node-level"):
        node_heatmap(res, mc)

    again = load(tmp_path / "out", out=tmp_path / "again")
    same = differential(again, "genotype", ("WT", "KO"), level="metaclusters")
    assert same["direction"].tolist() == mc["direction"].tolist()
    assert np.allclose(same["p"], mc["p"])
