"""peeters: FlowSOM on flow cytometry populations gated in ALN, and differential abundance
between conditions, drawn on the tree and as a node heatmap.

Named after Clara Peeters (1594 to after 1657), the Antwerp painter of still lifes.
"""

from .analysis import Result, inputs, load, load_model, run, save_model
from .differential import differential, plot_differential
from .gating import Prepared, load_gating, masks, read_fcs
from .heatmap import node_heatmap

__all__ = ["Prepared", "Result", "differential", "inputs", "load", "load_gating",
           "load_model", "masks", "node_heatmap", "plot_differential", "read_fcs", "run",
           "save_model"]
