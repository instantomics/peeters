"""A competitive chimera: every liver holds host, WT donor and EM3 donor cells, in
proportions that differ from mouse to mouse. One tree on all donor cells, each donor counted
on its own, and the paired comparison finds where EM3 fills a cell type worse or better
than WT in the same liver, whatever that liver's chimerism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_peeters import write_fcs

from peeters import differential, node_heatmap, plot_differential, run

CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "BV421-A", "AF700-A", "APC-A", "PE-A", "FITC-A"]
MARKERS = ["", "", "", "CD45-1", "CD45-2", "F4/80", "Tim4", "Clec2"]
#: F4/80, Tim4, Clec2 per cell type
TYPES = {"KC": (6000, 9000, 300), "moKC": (6000, 150, 7000), "mono": (100, 120, 200)}
WT = {"KC": 0.5, "moKC": 0.3, "mono": 0.2}
EM3 = {"KC": 0.1, "moKC": 0.5, "mono": 0.4}
#: (CD45.1, CD45.2): host CD45.1, WT donor CD45.1.2, EM3 donor CD45.2
CONGENIC = {"host": (8000, 50), "WT": (8000, 8000), "EM3": (60, 8000)}

GATING = """format: aln-gating/1
files:
{files}
compensation: none
samples: {{table: samples.csv, match: data}}
populations:
  Cells: {{parent: null, rectangle: {{x: FSC-A, y: SSC-A, x_min: 50000, y_min: 25000}}}}
  Donor: {{parent: Cells, range: {{x: CD45-2, min: 1000}}}}
  WT donor: {{parent: Donor, range: {{x: CD45-1, min: 1000}}}}
  EM3 donor: {{parent: Donor, range: {{x: CD45-1, max: 1000}}}}
"""


def liver(path, rng, n_wt: int, n_em3: int, n_host: int = 1500) -> dict:
    rows, truth = [], {}
    for who, n, shares in (("host", n_host, WT), ("WT", n_wt, WT), ("EM3", n_em3, EM3)):
        kinds = rng.choice(list(TYPES), n, p=[shares[k] for k in TYPES])
        truth[who] = n
        c1, c2 = CONGENIC[who]
        for k in kinds:
            f480, tim4, clec2 = (v * rng.lognormal(0, 0.15) for v in TYPES[k])
            fsc = rng.uniform(80000, 200000)
            rows.append([fsc, fsc * 0.8, rng.uniform(40000, 120000),
                         c1 * rng.lognormal(0, 0.1), c2 * rng.lognormal(0, 0.1),
                         f480, tim4, clec2])
    write_fcs(path, rng.permutation(np.array(rows)), CHANNELS, MARKERS, {"$CYT": "test"})
    return truth


@pytest.fixture(scope="module")
def chimeras(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp") / "exp" / "exp_chim000000"
    (root / "gating").mkdir(parents=True)
    ind = tmp_path_factory.mktemp("in")
    rng = np.random.default_rng(11)
    entries, samples, truths = [], [], {}
    for i in range(6):  # chimerism from mostly WT to mostly EM3
        n_wt, n_em3 = int(rng.integers(400, 1600)), int(rng.integers(400, 1600))
        name = f"Liver_{i + 1}.fcs"
        truths[name] = liver(ind / name, rng, n_wt, n_em3)
        entries.append(f"- {{data: dat_chim{i:06d}, name: {name}}}")
        samples.append({"data": f"dat_chim{i:06d}", "mouse": f"M{i + 1}",
                        "batch": "A" if i < 5 else "B"})
    (root / "gating" / "c.gating.yaml").write_text(GATING.format(files="\n".join(entries)))
    pd.DataFrame(samples).to_csv(root / "samples.csv", index=False)
    return root, ind, truths


def _run(chimeras, tmp_path):
    root, ind, _ = chimeras
    return run(root / "gating" / "c.gating.yaml", "Donor", ["F4/80", "Tim4", "Clec2"],
              parts={"WT": "WT donor", "EM3": "EM3 donor"}, in_dir=str(ind), out=tmp_path,
              per_file=800, xdim=5, ydim=5, metaclusters=3, seed=2, by="part", plots=False)


def test_parts_are_counted_on_one_tree(chimeras, tmp_path):
    truths = chimeras[2]
    res = _run(chimeras, tmp_path)
    f = res.files
    assert len(f) == 12 and set(f["part"]) == {"WT", "EM3"}
    for _, r in f.iterrows():  # each part's own cells; the donors together for the file
        t = truths[r["source_file"]]
        assert r["population_events"] == t[r["part"]]
        assert r["file_population_events"] == t["WT"] + t["EM3"]
    assert set(f["mouse"]) == {f"M{i}" for i in range(1, 7)}  # the samples table joined
    shares = res.wide().sum(axis=1)  # each part's shares are of its own cells
    assert np.allclose(shares, 100.0)


def test_paired_comparison_finds_where_em3_falls_short(chimeras, tmp_path):
    res = _run(chimeras, tmp_path)
    kc = res.medians["Tim4"].idxmax()
    mokc = res.medians["Clec2"].idxmax()
    mono = res.medians["F4/80"].idxmin()
    mc = differential(res, "part", ("WT", "EM3"), level="metaclusters", paired="source_file")
    d = mc.set_index("metacluster")
    assert (d.loc[kc, "direction"], d.loc[mokc, "direction"], d.loc[mono, "direction"]) == \
        ("down", "up", "up")
    assert d.loc[kc, "log2_fold"] == pytest.approx(np.log2(0.1 / 0.5), abs=0.35)
    assert d.loc[mono, "log2_fold"] == pytest.approx(1.0, abs=0.35)
    assert (d["pairs"] == 6).all() and d.loc[kc, "agree"] == 6
    assert mc.attrs["paired"] == "source_file" and mc.attrs["n_reference"] == 6

    # five pairs: no q can pass (p >= 0.0625); every pair the same way still calls it
    five = differential(res, "part", ("WT", "EM3"), level="metaclusters",
                        paired="source_file", where={"batch": "A"}, out=tmp_path / "a")
    assert (five["q"] >= 0.0625).all() and (five["direction"] == "").all()
    five = differential(res, "part", ("WT", "EM3"), level="metaclusters",
                        paired="source_file", where={"batch": "A"}, call="consistent",
                        out=tmp_path / "b")
    assert five.set_index("metacluster").loc[kc, "direction"] == "down"

    nodes = differential(res, "part", ("WT", "EM3"), paired="source_file", call="consistent")
    a = differential(res, "part", ("WT", "EM3"), paired="source_file", call="consistent",
                     where={"batch": "A"}, out=tmp_path / "c")
    assert node_heatmap(res, [nodes, a]).exists()
    assert plot_differential(res, nodes).exists()

    # with more cells asked of every pair than any node has, nothing is tested or called
    none = differential(res, "part", ("WT", "EM3"), paired="source_file", call="consistent",
                        min_events=10**6, out=tmp_path / "d")
    assert not none["tested"].any() and (none["direction"] == "").all()
    assert (nodes["fewest_events"] >= 0).all()

    with pytest.raises(ValueError, match="consistent"):
        differential(res, "part", ("WT", "EM3"), call="consistent")  # not paired
    with pytest.raises(ValueError, match="one WT and one EM3"):
        differential(res, "part", ("WT", "EM3"), paired="batch")  # not a pair per key
