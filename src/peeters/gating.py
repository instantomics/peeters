"""ALN gating files (``*.gating.yaml``, format aln-gating/1) applied in a run, so FlowSOM
clusters exactly the population people gated in ALN's editor.

A port of ALN's own flow gating engine (the flow plugin's gating/spec.py, scales.py,
events.py and engine.py): the same compensation, scales and gate rules. Ranges and
rectangles compare data units (compensated, off-scale events on the axis's edge;
``min <= x < max``); polygons compare display units, where they were drawn. Keep it in step
with the engine: ``gate_counts.csv`` of a run should equal the gating's counts in ALN
(``read_gating``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LN10 = math.log(10.0)
PARAMS: dict[str, dict[str, float | None]] = {
    "linear": {"min": 0.0, "max": None},
    "log": {"T": None, "M": 4.5},
    "logicle": {"T": None, "W": 0.5, "M": 4.5, "A": 0.0},
    "biex": {"neg": 0.0, "width": -10.0, "pos": 4.418540, "max": None},
    "asinh": {"T": None, "M": 4.0, "A": 0.0},
}
GATE_KINDS = ("polygon", "rectangle", "range", "boolean")


# --- scales ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scale:
    kind: str
    params: tuple

    @property
    def p(self) -> dict[str, float]:
        return dict(self.params)


def parse_scale(value: Any, top: float = 262144.0) -> Scale:
    if isinstance(value, str):
        kind, given = value, {}
    elif isinstance(value, dict) and len(value) == 1:
        kind, given = next(iter(value.items()))
        given = given or {}
    else:
        raise ValueError(f"not a scale: {value!r}")
    if kind not in PARAMS:
        raise ValueError(f"unknown scale {kind!r}")
    params = {n: float(given.get(n, top if d is None else d)) for n, d in PARAMS[kind].items()}
    return Scale(kind, tuple(sorted(params.items())))


def default_scale(channel: str, top: float = 262144.0) -> Scale:
    """Scatter and time linear, fluorescence logicle (as ALN)."""
    up = channel.upper()
    if up.startswith(("FSC", "SSC")) or up in ("TIME", "EVENT_COUNT", "EVENT"):
        return Scale("linear", (("max", float(top)), ("min", 0.0)))
    return Scale("logicle", (("A", 0.0), ("M", 4.5), ("T", float(top)), ("W", 0.5)))


class Linear:
    def __init__(self, lo, hi):
        self.lo, self.span = lo, hi - lo

    def __call__(self, x):
        return (np.asarray(x, dtype=np.float64) - self.lo) / self.span

    def inverse(self, y):
        return np.asarray(y, dtype=np.float64) * self.span + self.lo


class Log:
    def __init__(self, T, M):  # noqa: N803
        self.T, self.M = T, M

    def __call__(self, x):
        x = np.maximum(np.asarray(x, dtype=np.float64), self.T * 10.0 ** (-self.M) / 10)
        return np.log10(x / self.T) / self.M + 1

    def inverse(self, y):
        return self.T * 10.0 ** ((np.asarray(y, dtype=np.float64) - 1) * self.M)


class Asinh:
    def __init__(self, T, M, A):  # noqa: N803
        self.M, self.A = M, A
        self.k = math.sinh(M * LN10) / T

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        return (np.arcsinh(x * self.k) + self.A * LN10) / ((self.M + self.A) * LN10)

    def inverse(self, y):
        y = np.asarray(y, dtype=np.float64)
        return np.sinh(y * (self.M + self.A) * LN10 - self.A * LN10) / self.k


class Logicle:
    """Parks et al. 2006, as ALN's (checked there against flowutils)."""

    def __init__(self, T, W, M, A):  # noqa: N803
        self.T = T
        w = W / (M + A)
        x2 = A / (M + A)
        self.x1 = x2 + w
        x0 = x2 + 2 * w
        self.b = (M + A) * LN10
        self.d = self._solve_d(self.b, w)
        c_a = math.exp(x0 * (self.b + self.d))
        mf_a = math.exp(self.b * self.x1) - c_a / math.exp(self.d * self.x1)
        self.a = T / ((math.exp(self.b) - mf_a) - c_a / math.exp(self.d))
        self.c = c_a * self.a
        self.f = -mf_a * self.a

    @staticmethod
    def _solve_d(b, w):
        if w == 0:
            return b
        lo, hi = 1e-300, b
        for _ in range(200):
            mid = (lo + hi) / 2
            if 2 * (math.log(mid) - math.log(b)) + w * (b + mid) < 0:
                lo = mid
            else:
                hi = mid
        d = (lo + hi) / 2
        for _ in range(5):
            g = 2 * (math.log(d) - math.log(b)) + w * (b + d)
            d -= g / (2 / d + w)
        return d

    def _biex(self, y):
        return self.a * np.exp(self.b * y) - self.c * np.exp(-self.d * y) + self.f

    def inverse(self, y):
        y = np.asarray(y, dtype=np.float64)
        neg = y < self.x1
        yy = np.where(neg, 2 * self.x1 - y, y)
        x = self._biex(yy)
        return np.where(neg, -x, x)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        ax = np.abs(x)
        grid = np.linspace(self.x1, 1.5, 4097)
        top = float(self._biex(np.array(1.5)))
        y = np.interp(ax, self._biex(grid), grid)
        y = np.where(ax > top, 1.5 + np.log(np.maximum(ax, 1) / top) / self.b, y)
        for _ in range(4):
            fx = self._biex(y) - ax
            dfx = self.a * self.b * np.exp(self.b * y) + self.c * self.d * np.exp(-self.d * y)
            y = y - fx / dfx
        return np.where(x < 0, 2 * self.x1 - y, y)


def _log_root(b, w):
    x_lo, x_hi = 0.0, b
    d = (x_lo + x_hi) / 2
    dx_last = abs(int(x_lo - x_hi))
    fb = -2 * math.log(b) + w * b
    f = 2.0 * math.log(d) + w * b + fb
    df = 2 / d + w
    if w == 0:
        return b
    for _ in range(100):
        if (((d - x_hi) * df - f) - ((d - x_lo) * df - f)) > 0 or abs(2 * f) > abs(dx_last * df):
            dx = (x_hi - x_lo) / 2
            d = x_lo + dx
            if d == x_lo:
                return d
        else:
            dx = f / df
            t = d
            d -= dx
            if d == t:
                return d
        if abs(dx) < 1.0e-12:
            return d
        dx_last = dx
        f = 2 * math.log(d) + w * d + fb
        df = 2 / d + w
        if f < 0:
            x_lo = d
        else:
            x_hi = d
    return d


class Biex:
    """FlowJo 10's biex lookup table (as ALN, which ported it from FlowKit, BSD)."""

    def __init__(self, neg, width, pos, top, channel_range=4096):
        decades = pos
        width_log = math.log10(-width)
        decades = decades - width_log / 2
        extra = max(neg, 0.0) + width_log / 2
        zero_point = int((extra * channel_range) / (extra + decades))
        zero_point = int(min(zero_point, channel_range / 2))
        if zero_point > 0:
            decades = extra * channel_range / zero_point
        w = width_log / (2 * decades)
        positive_range = LN10 * decades
        minimum = top / math.exp(positive_range)
        negative_range = _log_root(positive_range, w)
        n = channel_range + 1
        values = np.arange(n, dtype=np.float64)
        positive = np.exp(values / float(n) * positive_range)
        negative = np.exp(values / float(n) * -negative_range)
        s = math.exp((positive_range + negative_range) * (w + extra / decades))
        negative *= s
        s = positive[zero_point] - negative[zero_point]
        positive[zero_point:n] = positive[zero_point:n] - negative[zero_point:n]
        positive[zero_point:n] = minimum * (positive[zero_point:n] - s)
        below = np.arange(zero_point)
        positive[below] = -positive[2 * zero_point - below]
        self.x, self.y = positive, values / channel_range

    def __call__(self, x):
        return np.interp(np.asarray(x, dtype=np.float64), self.x, self.y)

    def inverse(self, y):
        return np.interp(np.asarray(y, dtype=np.float64), self.y, self.x)


_made: dict[Scale, Any] = {}


def transform(s: Scale):
    t = _made.get(s)
    if t is None:
        p = s.p
        t = {"linear": lambda: Linear(p["min"], p["max"]),
             "log": lambda: Log(p["T"], p["M"]),
             "logicle": lambda: Logicle(p["T"], p["W"], p["M"], p["A"]),
             "biex": lambda: Biex(p["neg"], p["width"], p["pos"], p["max"]),
             "asinh": lambda: Asinh(p["T"], p["M"], p["A"])}[s.kind]()
        _made[s] = t
    return t


# --- FCS files ---------------------------------------------------------------------------


def _text(raw: bytes) -> dict[str, str]:
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return {}
    delim, items, cur, i = text[0], [], [], 1
    while i < len(text):
        ch = text[i]
        if ch == delim:
            if i + 1 < len(text) and text[i + 1] == delim:
                cur.append(delim)
                i += 2
                continue
            items.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    if cur:
        items.append("".join(cur))
    return {items[k].strip().upper(): items[k + 1].strip() for k in range(0, len(items) - 1, 2)}


def _spillover(kw: dict[str, str]):
    raw = next((kw[k] for k in ("$SPILLOVER", "$SPILL", "SPILL", "SPILLOVER") if kw.get(k)), None)
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    try:
        n = int(parts[0])
        names = parts[1:1 + n]
        values = [float(v) for v in parts[1 + n:1 + n + n * n]]
    except (ValueError, IndexError):
        return None
    if n < 1 or len(names) != n or len(values) != n * n:
        return None
    return names, np.array(values, dtype=np.float64).reshape(n, n)


@dataclass
class Events:
    path: str
    channels: list[str]
    markers: list[str]
    ranges: list[float]
    data: np.ndarray  # events x channels, float32, as stored
    spillover: Any
    keywords: dict[str, str]

    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    def index(self, name: str) -> int:
        """By detector ($PnN), else by marker ($PnS, or its part before '#'), without case;
        FlowJo's "Comp-" prefix ignored (as ALN)."""
        name = name.strip()
        bare = name[5:] if name.lower().startswith("comp-") else name
        for cand in (name, bare):
            if cand in self.channels:
                return self.channels.index(cand)
        low = bare.lower()
        for i, c in enumerate(self.channels):
            if c.lower() == low:
                return i
        for i, m in enumerate(self.markers):
            if m and (m.lower() == low or m.split("#")[0].strip().lower() == low):
                return i
        said = ", ".join(f"{c} ({m})" if m else c for c, m in zip(self.channels, self.markers))
        raise KeyError(f"no channel {name!r} in {Path(self.path).name}; it has {said}")

    def label(self, i: int) -> str:
        return self.markers[i] or self.channels[i]


def read_fcs(path: str | Path) -> Events:
    """An FCS 2.0/3.0/3.1 list-mode file, read as ALN reads it."""
    with open(path, "rb") as fh:
        head = fh.read(58)
        if len(head) < 58 or not re.match(rb"FCS\d\.\d", head[:6]):
            raise ValueError(f"{path}: not an FCS file")
        offs = [int(head[10 + 8 * k:18 + 8 * k].strip() or 0) for k in range(4)]
        fh.seek(offs[0])
        kw = _text(fh.read(offs[1] - offs[0] + 1))
        begin = int(kw.get("$BEGINDATA", "0").strip() or 0) or offs[2]
        end = int(kw.get("$ENDDATA", "0").strip() or 0) or offs[3]
        n_par, n_tot = int(kw["$PAR"]), int(kw["$TOT"])
        order = "<" if kw.get("$BYTEORD", "1,2,3,4").replace(" ", "").startswith("1") else ">"
        dtype = kw.get("$DATATYPE", "F").upper()
        fh.seek(begin)
        raw = fh.read(end - begin + 1)
    if dtype in ("F", "D"):
        width = 4 if dtype == "F" else 8
        data = np.frombuffer(raw[:n_par * n_tot * width],
                             dtype=f"{order}{'f4' if dtype == 'F' else 'f8'}")
        data = data.reshape(n_tot, n_par).astype(np.float32)
    elif dtype == "I":
        bits = int(kw.get("$P1B", "16"))
        width = bits // 8
        ints = np.frombuffer(raw[:n_par * n_tot * width], dtype=f"{order}u{width}")
        ints = ints.reshape(n_tot, n_par)
        data = np.empty((n_tot, n_par), dtype=np.float32)
        for i in range(n_par):
            col = ints[:, i]
            rng = int(float(kw.get(f"$P{i + 1}R", "0") or 0))
            if rng and (rng & (rng - 1)) == 0:
                col = col % rng
            f1, _, f2 = (kw.get(f"$P{i + 1}E", "0,0") or "0,0").partition(",")
            dec, off = float(f1 or 0), float(f2 or 0)
            if dec > 0 and rng:
                data[:, i] = (off or 1.0) * 10 ** (dec * col.astype(np.float64) / rng)
            else:
                data[:, i] = col
    else:
        raise ValueError(f"{path}: FCS $DATATYPE {dtype} is not read")
    channels = [kw.get(f"$P{i}N", f"P{i}") for i in range(1, n_par + 1)]
    markers = [kw.get(f"$P{i}S", "") for i in range(1, n_par + 1)]
    markers = ["" if m == c else m for m, c in zip(markers, channels)]
    ranges = []
    for i in range(1, n_par + 1):
        try:
            ranges.append(float(kw.get(f"$P{i}R", "262144")))
        except ValueError:
            ranges.append(262144.0)
    return Events(str(path), channels, markers, ranges, data, _spillover(kw), kw)


# --- the gating file ---------------------------------------------------------------------


@dataclass
class Population:
    name: str
    parent: str | None
    kind: str
    gate: dict[str, Any]
    exclude: list[str] = field(default_factory=list)


@dataclass
class FileEntry:
    data: str
    name: str | None = None
    sample: str | None = None
    gates: dict[str, tuple[str, dict]] = field(default_factory=dict)
    compensation: Any = None
    scales: dict[str, Any] = field(default_factory=dict)


@dataclass
class Gating:
    path: str
    title: str
    files: list[FileEntry]
    compensation: Any
    scales: dict[str, Any]
    populations: dict[str, Population]
    samples: dict | None = None
    groups: str | None = None

    def gate_for(self, name: str, entry: FileEntry | None):
        if entry is not None and name in entry.gates:
            return entry.gates[name]
        p = self.populations[name]
        return p.kind, p.gate

    def entry(self, data: str) -> FileEntry:
        for f in self.files:
            if f.data == data:
                return f
        raise KeyError(f"{data} is not one of the gating's files")


def _gate_of(body: dict) -> tuple[str, dict]:
    kinds = [k for k in GATE_KINDS if k in body]
    if len(kinds) != 1:
        raise ValueError(f"give exactly one gate of {GATE_KINDS}: {body}")
    g = dict(body[kinds[0]])
    if kinds[0] == "boolean" and isinstance(g.get("of"), str):
        g["of"] = [g["of"]]
    return kinds[0], g


def load_gating(path: str | Path) -> Gating:
    """A gating file as ALN writes it (in a run: ``/exp/<id>/gating/<name>.gating.yaml``)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if raw.get("format", "aln-gating/1") != "aln-gating/1":
        raise ValueError(f"unknown gating format {raw.get('format')!r}")
    files = []
    for f in raw.get("files") or []:
        if isinstance(f, str):
            f = {"data": f}
        e = FileEntry(data=f["data"].strip().removeprefix("[[").removesuffix("]]"),
                      name=f.get("name"), sample=f.get("sample"),
                      compensation=f.get("compensation"), scales=f.get("scales") or {})
        for name, body in (f.get("gates") or {}).items():
            e.gates[str(name)] = _gate_of(body)
        files.append(e)
    pops = {}
    for name, body in (raw.get("populations") or {}).items():
        parent = body.get("parent")
        parent = None if parent in (None, "", "root") else str(parent)
        kind, gate = _gate_of(body)
        ex = body.get("exclude") or []
        pops[str(name).strip()] = Population(str(name).strip(), parent, kind, gate,
                                             [ex] if isinstance(ex, str) else list(ex))
    samples = raw.get("samples")
    if isinstance(samples, str):
        samples = {"table": samples}
    return Gating(str(path), str(raw.get("title") or ""), files,
                  raw.get("compensation", "file"), raw.get("scales") or {}, pops,
                  samples, raw.get("groups"))


# --- one file, gated ---------------------------------------------------------------------


class Prepared:
    """One FCS file with the gating's compensation and scales (ALN's engine.Prepared)."""

    def __init__(self, g: Gating | None, entry: FileEntry | None, ev: Events):
        self.g, self.entry, self.ev = g, entry, ev
        comp_value = "file"
        if g is not None:
            comp_value = entry.compensation if entry and entry.compensation is not None \
                else g.compensation
        self.comp = None
        m = None
        if comp_value == "file":
            m = ev.spillover
        elif isinstance(comp_value, dict):
            m = comp_value["channels"], np.array(comp_value["matrix"], dtype=np.float64)
        if m is not None:
            names, matrix = m
            self.comp = ([ev.index(n) for n in names], np.linalg.inv(matrix))
        self.scales: dict[int, Scale] = {}
        for source in ((g.scales if g else {}), (entry.scales if entry else {})):
            for ch, value in source.items():
                try:
                    i = ev.index(ch)
                except KeyError:
                    continue
                self.scales[i] = parse_scale(value, ev.ranges[i])
        self._cache: dict[tuple, np.ndarray] = {}

    def channel(self, name: str) -> int:
        return self.ev.index(name)

    def scale(self, i: int) -> Scale:
        return self.scales.get(i) or default_scale(self.ev.channels[i], self.ev.ranges[i])

    def data(self, i: int) -> np.ndarray:
        """Channel i compensated (data units), float32."""
        if self.comp is None or i not in self.comp[0]:
            return self.ev.data[:, i]
        key = ("data", i)
        if key not in self._cache:
            idx, inv = self.comp
            col = idx.index(i)
            out = np.zeros(self.ev.n, dtype=np.float64)
            for row, j in enumerate(idx):
                w = inv[row, col]
                if w != 0:
                    out += self.ev.data[:, j].astype(np.float64) * w
            self._cache[key] = out.astype(np.float32)
        return self._cache[key]

    def display(self, i: int) -> np.ndarray:
        """Channel i on its scale, 0 to 1 (events beyond the axis on its edge)."""
        key = ("display", i)
        if key not in self._cache:
            self._cache[key] = np.clip(transform(self.scale(i))(self.data(i)), 0.0, 1.0) \
                .astype(np.float32)
        return self._cache[key]

    def gated(self, i: int) -> np.ndarray:
        key = ("gated", i)
        if key not in self._cache:
            lo, hi = (float(v) for v in transform(self.scale(i)).inverse(np.array([0.0, 1.0])))
            self._cache[key] = np.clip(self.data(i), lo, hi).astype(np.float32)
        return self._cache[key]

    def drop(self) -> None:
        self._cache.clear()


def _in_polygon(x, y, pts):
    inside = np.zeros(x.shape, dtype=bool)
    px, py = pts[:, 0], pts[:, 1]
    j = len(px) - 1
    for i in range(len(px)):
        yi, yj = py[i], py[j]
        if yi != yj:
            crosses = (yi > y) != (yj > y)
            xs = (px[j] - px[i]) * (y - yi) / (yj - yi) + px[i]
            inside ^= crosses & (x < xs)
        j = i
    return inside


def _inside(p: Prepared, kind: str, gate: dict, where: np.ndarray, masks: dict) -> np.ndarray:
    if kind == "range":
        v = p.gated(p.channel(gate["x"]))[where]
        m = np.ones(len(where), dtype=bool)
        if gate.get("min") is not None:
            m &= v >= gate["min"]
        if gate.get("max") is not None:
            m &= v < gate["max"]
        return m
    if kind == "rectangle":
        m = np.ones(len(where), dtype=bool)
        for a in ("x", "y"):
            lo, hi = gate.get(f"{a}_min"), gate.get(f"{a}_max")
            if lo is None and hi is None:
                continue
            v = p.gated(p.channel(gate[a]))[where]
            if lo is not None:
                m &= v >= lo
            if hi is not None:
                m &= v < hi
        return m
    if kind == "polygon":
        ix, iy = p.channel(gate["x"]), p.channel(gate["y"])
        pts = np.array(gate["points"], dtype=np.float64)
        dpts = np.column_stack([transform(p.scale(ix))(pts[:, 0]),
                                transform(p.scale(iy))(pts[:, 1])])
        return _in_polygon(p.display(ix)[where].astype(np.float64),
                           p.display(iy)[where].astype(np.float64), dpts)
    if kind == "boolean":
        parts = [masks[o][where] for o in gate["of"]]
        if gate["op"] == "and":
            return np.logical_and.reduce(parts)
        if gate["op"] == "or":
            return np.logical_or.reduce(parts)
        return ~parts[0]
    raise ValueError(f"unknown gate kind {kind}")


def masks(p: Prepared, only: set[str] | None = None) -> dict[str, np.ndarray]:
    """Population name -> boolean mask over all events (``only``: those and what they need)."""
    g = p.g
    need = None
    if only is not None:
        need, todo = set(), list(only)
        while todo:
            n = todo.pop()
            if n in need:
                continue
            if n not in g.populations:
                raise KeyError(f"no population {n!r} in {Path(g.path).name}; it has "
                               f"{', '.join(g.populations)}")
            need.add(n)
            pop = g.populations[n]
            todo += ([pop.parent] if pop.parent else []) + pop.exclude
            if pop.kind == "boolean":
                todo += pop.gate["of"]
    out: dict[str, np.ndarray] = {}
    every = np.ones(p.ev.n, dtype=bool)
    for name, pop in g.populations.items():
        if need is not None and name not in need:
            continue
        parent = out[pop.parent] if pop.parent else every
        where = np.flatnonzero(parent)
        kind, gate = g.gate_for(name, p.entry)
        inside = _inside(p, kind, gate, where, out)
        if gate.get("outside"):
            inside = ~inside
        m = np.zeros(p.ev.n, dtype=bool)
        m[where[inside]] = True
        for ex in pop.exclude:
            if ex in out:
                m &= ~out[ex]
        out[name] = m
    return out
