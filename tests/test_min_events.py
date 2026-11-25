"""min_events: a share needs cells behind it. A node one group lost is still tested; a node
with a handful of cells everywhere is not, and the correction runs without it."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from peeters import differential


class FakeRun:
    """What differential reads of a run: the files, and counts per file and node."""

    def __init__(self, counts: dict[str, dict[int, int]], groups: dict[str, str], out: Path):
        self.files = pd.DataFrame({"file": list(groups), "genotype": list(groups.values())})
        rows = []
        for f, per_node in counts.items():
            total = sum(per_node.values())
            for node, n in per_node.items():
                rows.append({"file": f, "node": node, "metacluster": node, "count": n,
                             "percent": 100.0 * n / total})
        self.nodes = pd.DataFrame(rows)
        self.metaclusters = self.nodes.copy()
        self.out = out

    def wide(self, level: str = "metaclusters", value: str = "percent") -> pd.DataFrame:
        col = "metacluster" if level == "metaclusters" else "node"
        long = self.metaclusters if level == "metaclusters" else self.nodes
        return long.pivot(index="file", columns=col, values=value)


def test_lost_in_one_group_is_tested_a_handful_is_not(tmp_path):
    groups = {f"wt{i}": "WT" for i in range(4)} | {f"ko{i}": "KO" for i in range(4)}
    counts = {}
    for i in range(4):
        # node 1: plenty in WT, none in KO; node 2: 5 cells everywhere; node 3: the rest
        counts[f"wt{i}"] = {1: 400 + 10 * i, 2: 5, 3: 2000}
        counts[f"ko{i}"] = {1: 0, 2: 5 + i, 3: 2000 + 10 * i}
    d = differential(FakeRun(counts, groups, tmp_path), "genotype", ("WT", "KO"))
    d = d.set_index("node")
    assert bool(d.loc[1, "tested"]) and d.loc[1, "direction"] == "down"
    assert not bool(d.loc[2, "tested"]) and d.loc[2, "direction"] == ""
    assert d.loc[2, "fewest_events"] == 5
    assert d.attrs["min_events"] == 20
    lax = differential(FakeRun(counts, groups, tmp_path), "genotype", ("WT", "KO"),
                       min_events=0).set_index("node")
    assert bool(lax.loc[2, "tested"])
