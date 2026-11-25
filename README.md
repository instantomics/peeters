# Peeters

FlowSOM on a gated flow cytometry population over many FCS files, and differential
abundance between conditions, drawn on the FlowSOM tree and as a node heatmap.

Named after Clara Peeters (1594 to after 1657), the Antwerp painter of still lifes.

```bash
pip install peeters
```

## What it does

1. **Gate** each file as an [ALN](https://github.com/guilliottslab) gating file
   (`*.gating.yaml`, format `aln-gating/1`) says: the same compensation, scales and gates
   as ALN's own engine, so the counts are ALN's counts. Files that are already gated can be
   given without a gating.
2. **Cluster** the population with [FlowSOM](https://github.com/saeyslab/FlowSOM_Python):
   trained on the same number of cells from every file, then every cell of every file mapped
   onto the tree.
3. **Count** each file's share of every node and metacluster, and the medians of every marker.
4. **Compare** two groups of files per node or metacluster, as FlowSOM's `GroupStats` does:
   per-file shares, Mann-Whitney test, Benjamini-Hochberg.
5. **Draw** it: the tree with a coloured band around every node that differs, and a node
   heatmap with each node as high as its share and a strip per comparison.

## Use

```python
from peeters import run, differential, plot_differential, node_heatmap

res = run("gating/liver.gating.yaml", "CD45+",
          markers=["CD11b", "F4/80", "CD64", "Tim4", "Clec4f", "Vsig4", "Clec2", "Ly6C"],
          extra=["CD45.1", "CD45.2"],          # reported per metacluster, not clustered on
          samples="samples.csv", by="genotype", metaclusters=12, seed=42, out="flowsom")

nodes = differential(res, "genotype", ("WT", "KO"))                  # reference first
mcs = differential(res, "genotype", ("WT", "KO"), level="metaclusters")
plot_differential(res, nodes)                                        # the tree with bands

exp1 = differential(res, "genotype", ("WT", "KO"), where={"cohort": "Exp1"})
exp2 = differential(res, "genotype", ("WT", "KO"), where={"cohort": "Exp2"})
node_heatmap(res, [exp1, exp2], names={6: "KC-like"})                # a strip per cohort
```

`run` writes `files.csv`, `metaclusters.csv`, `nodes.csv`, `metacluster_medians.csv`,
`file_medians.csv`, `params.json`, `model.pkl` and the tree, heatmap and composition
figures; `load(folder)` reads a run back without training again.

## Citing

Every release is archived on Zenodo with a DOI of its own: cite the version you used (the
DOI of a version is on its Zenodo page and in the GitHub release), and FlowSOM:

- Van Gassen S, et al. FlowSOM: Using self-organizing maps for visualization and
  interpretation of cytometry data. *Cytometry A* 87, 636-645 (2015).
  [doi:10.1002/cyto.a.22625](https://doi.org/10.1002/cyto.a.22625)
- Quintelier K, et al. Analyzing high-dimensional cytometry data using FlowSOM.
  *Nat Protoc* 16, 3775-3801 (2021).
  [doi:10.1038/s41596-021-00550-0](https://doi.org/10.1038/s41596-021-00550-0)

GitHub's *Cite this repository* (from `CITATION.cff`) gives the same.

## Release

`./dist.sh <version>` from a clean, pushed `main`: tags, builds, makes the GitHub release
and publishes to PyPI (the token in `~/.secrets/pypi.env`).
