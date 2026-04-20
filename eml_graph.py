"""
Matplotlib-based visual display for the EML rewrite universe.

Loads the rewrite rules from `eml_universe` and renders each step in an
interactive matplotlib window with `← Back` and `Next →` buttons.

By default node labels use SymPy's plain string form (fast). Pass --latex to
use matplotlib mathtext + LaTeX from sympy (slow on large graphs).

Run:
    source venv/bin/activate && python3 eml_graph.py
    source venv/bin/activate && python3 eml_graph.py --latex --random

You will be asked **[d]eterministic**, **[r]andom**, or **[m]erge** unless
you pass ``--deterministic``, ``--random`` or ``--merge``.

Scroll wheel (or trackpad scroll) over the plot zooms in/out around the cursor.
Keys + / - zoom centered on the view (helps if scroll events do not fire).
Use the toolbar Pan tool to drag the view after zooming.

SymPy's ``log`` is the natural logarithm (same as ln); plain labels show ``ln``;
``--latex`` uses ``\\ln`` in the rendered math.

Edges are drawn as separate arcs so links are easier to tell apart. Node boxes
use a **semi-transparent** fill so links passing under other nodes stay visible.
Fonts and edge stroke widths scale when you zoom (like zooming a picture).
"""

import argparse
import math
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import symengine as sp
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.widgets import Button

from eml_dimension import compute_dimension, compute_dimension_from_center
from eml_universe import GraphState


# Dark theme (figure + nodes + edges + buttons)
DARK_BG = "#12141c"
NODE_FACE = "#2a334d"
NODE_FACE_ALPHA = 0.10
NODE_EDGE = "#6b9fe8"
NODE_TEXT = "#e6eaf2"
EDGE_COLOR = "#8fb6ee"
EDGE_ALPHA = 0.78
EDGE_WIDTH = 1.25
TITLE_COLOR = "#c8d0e0"
BTN_FACE = "#2d3748"
BTN_HOVER = "#3d4a5c"
BTN_LABEL = "#e6eaf2"
BTN_EDGE = "#4a5568"


# Recognised significant constants get a distinctive box color so they pop
# against the regular pale-blue nodes. Keys must be SymEngine-hashable Basics
# whose `==` against the corresponding node value returns True.
from symengine import E as _SYM_E, I as _SYM_I, Integer as _SYM_INT, pi as _SYM_PI

CONSTANT_FACES: Dict[sp.Basic, str] = {
    _SYM_INT(0):  "#cfd2cf",
    _SYM_INT(1):  "#ffd166",
    _SYM_INT(-1): "#ef476f",
    _SYM_E:       "#f96e46",
    _SYM_PI:      "#4cc9f0",
    _SYM_I:       "#b388eb",
}

CONSTANT_LABELS: Dict[sp.Basic, str] = {
    _SYM_INT(0):  "0",
    _SYM_INT(1):  "1",
    _SYM_INT(-1): "-1",
    _SYM_E:       "e",
    _SYM_PI:      "π",
    _SYM_I:       "i",
}


def constant_face_color(value: sp.Basic) -> Optional[str]:
    """Return the highlight color for `value` if it is one of the recognised
    significant constants, else None."""
    return CONSTANT_FACES.get(value)


def node_bbox_kwargs(font_size: float) -> Dict[str, object]:
    """Semi-transparent face so edges are not hidden by opaque boxes over the graph."""
    pad = 0.22 + 0.025 * font_size
    return dict(
        boxstyle=f"round,pad={pad}",
        fc=mcolors.to_rgba(NODE_FACE, alpha=NODE_FACE_ALPHA),
        ec=mcolors.to_rgba(NODE_EDGE, alpha=0.92),
        linewidth=max(0.45, font_size / 6.5),
    )


# ----------------------------- toolbar / nav buttons ----------------


def style_nav_button_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BTN_FACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def outline_matplotlib_button(btn: Button) -> None:
    """Matplotlib 3.10+ `Button` has no `.rect`; the face is `btn.ax.patch`."""
    btn.ax.patch.set_edgecolor(BTN_EDGE)
    btn.ax.patch.set_linewidth(0.8)


def style_matplotlib_toolbar(fig: plt.Figure) -> None:
    """Darken the default matplotlib navigation toolbar when the backend exposes it."""
    tub = getattr(fig.canvas, "toolbar", None)
    if tub is None:
        return
    # TkAgg / Tk backends
    if hasattr(tub, "winfo_toplevel"):
        try:
            root = tub.winfo_toplevel()
            root.configure(bg=DARK_BG)
            tub.configure(bg=DARK_BG)
            for child in tub.winfo_children():
                try:
                    child.configure(
                        bg=BTN_FACE,
                        fg=BTN_LABEL,
                        highlightbackground=DARK_BG,
                        activebackground=BTN_HOVER,
                    )
                except Exception:
                    pass
            return
        except Exception:
            pass
    # Qt backends
    try:
        win = fig.canvas.manager.window
        win.setStyleSheet(
            f"QWidget {{ background-color: {DARK_BG}; }}"
            f"QToolBar {{ background-color: {DARK_BG}; border: none; spacing: 2px; }}"
            f"QToolButton {{ background-color: {BTN_FACE}; color: {BTN_LABEL}; "
            f"border: 1px solid {BTN_EDGE}; border-radius: 3px; padding: 2px; }}"
            f"QToolButton:hover {{ background-color: {BTN_HOVER}; }}"
        )
    except Exception:
        pass


# ----------------------------- layout ------------------------------


def build_nx_graph(state: GraphState) -> nx.Graph:
    g = nx.Graph()
    for nid in state.values:
        g.add_node(nid)
    for link in state.edges:
        a, b = tuple(link)
        g.add_edge(a, b)
    return g


def compute_layout(g: nx.Graph, state: GraphState) -> Dict[int, tuple]:
    if len(state.values) == 1:
        only = next(iter(state.values))
        return {only: (0.0, 0.0)}
    if len(state.values) == 2:
        a, b = list(state.values)
        return {a: (-0.5, 0.0), b: (0.5, 0.0)}
    try:
        return nx.kamada_kawai_layout(g)
    except Exception:
        k = 3.0 / max(1, len(state.values) ** 0.3)
        return nx.spring_layout(g, seed=42, k=k, iterations=500)


# Data-coordinate node box dimensions. Boxes sit in data space so zooming is
# a literal picture zoom: everything scales together, boxes never overlap after
# zoom if they do not overlap before zoom.
CHAR_H_DATA = 0.14   # data-units for the rendered-text height (1em line)
BOX_PAD_DATA = 0.05  # padding between text and box edge (data coords)
BOX_GAP_DATA = 0.04  # extra gap enforced between two different boxes
# Measure aspects at the smallest fontsize we will ever actually render (same as
# MIN_READABLE_FS defined below). At tiny fontsizes matplotlib's integer pixel
# rounding widens glyphs relative to scaled-down 100pt measurements, so sizing
# boxes from a 100pt reference leaves them systematically too narrow at display
# fontsizes of 4-10pt. Measuring at the minimum rendered fontsize gives us the
# worst-case aspect and therefore the widest (safest) box.
LABEL_REF_FS_PT = 4.0


# pixel-aspect (width/height) of each label at LABEL_REF_FS_PT, keyed by string
_LABEL_ASPECT_CACHE: Dict[str, float] = {}


def measure_label_aspects(
    labels: List[str], fig: plt.Figure
) -> Dict[str, float]:
    """Render each unique label once to measure its true width/height ratio in
    pixels at a fixed reference font size; cache across calls."""
    unique = [s for s in set(labels) if s not in _LABEL_ASPECT_CACHE]
    if unique:
        tmp = fig.add_axes([0.0, 0.0, 1e-3, 1e-3])
        tmp.set_axis_off()
        tmp.patch.set_alpha(0.0)
        artists = [
            tmp.text(0, 0, s, fontsize=LABEL_REF_FS_PT, ha="left", va="baseline")
            for s in unique
        ]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for s, t in zip(unique, artists):
            bb = t.get_window_extent(renderer=renderer)
            if bb.height > 0 and bb.width > 0:
                _LABEL_ASPECT_CACHE[s] = bb.width / bb.height
            else:
                _LABEL_ASPECT_CACHE[s] = max(1.0, 0.55 * len(s))
        tmp.remove()
    return {s: _LABEL_ASPECT_CACHE[s] for s in set(labels)}


def estimate_box_size_data(label: str, aspect: float) -> Tuple[float, float]:
    """Box is sized from the label's measured pixel aspect so text always fits."""
    w = aspect * CHAR_H_DATA + 2 * BOX_PAD_DATA
    h = CHAR_H_DATA + 2 * BOX_PAD_DATA
    return w, h


def scale_layout_for_boxes(
    pos: Dict[int, List[float]],
    sizes: Dict[int, Tuple[float, float]],
) -> Dict[int, List[float]]:
    """Spread the base layout so boxes have room before rigorous packing."""
    n = len(pos)
    if n <= 1:
        return pos
    avg_box = sum(max(w, h) for w, h in sizes.values()) / max(1, n)
    scale = 1.6 * avg_box * max(2.0, n ** 0.5)
    return {nid: [p[0] * scale, p[1] * scale] for nid, p in pos.items()}


def pack_rectangles_no_overlap(
    pos: Dict[int, List[float]],
    sizes: Dict[int, Tuple[float, float]],
    gap: float = BOX_GAP_DATA,
    max_iters: int = 800,
) -> Dict[int, List[float]]:
    """Rigorous rectangle separation: shifts nodes along the axis of minimum
    overlap until no pair of rectangles intersect (or max_iters reached)."""
    ids = list(pos.keys())
    n = len(ids)
    if n <= 1:
        return pos
    p = np.array([pos[nid] for nid in ids], dtype=float)
    sz = np.array([sizes[nid] for nid in ids], dtype=float)
    half_w = sz[:, 0] / 2.0
    half_h = sz[:, 1] / 2.0
    for _ in range(max_iters):
        dx = p[:, 0][:, None] - p[:, 0][None, :]
        dy = p[:, 1][:, None] - p[:, 1][None, :]
        wsum = half_w[:, None] + half_w[None, :] + gap
        hsum = half_h[:, None] + half_h[None, :] + gap
        ox = wsum - np.abs(dx)
        oy = hsum - np.abs(dy)
        overlap = (ox > 0) & (oy > 0)
        np.fill_diagonal(overlap, False)
        if not overlap.any():
            break
        along_x = overlap & (ox <= oy)
        along_y = overlap & ~along_x
        sgn_x = np.where(dx >= 0, 1.0, -1.0)
        sgn_y = np.where(dy >= 0, 1.0, -1.0)
        push_x = np.where(along_x, (ox / 2.0) * sgn_x, 0.0)
        push_y = np.where(along_y, (oy / 2.0) * sgn_y, 0.0)
        p[:, 0] += 0.55 * push_x.sum(axis=1)
        p[:, 1] += 0.55 * push_y.sum(axis=1)
    return {nid: [float(p[i, 0]), float(p[i, 1])] for i, nid in enumerate(ids)}


def compute_view_bounds(
    pos: Dict[int, List[float]],
    sizes: Dict[int, Tuple[float, float]],
    margin: float = 0.08,
) -> Tuple[float, float, float, float]:
    xs0 = [pos[nid][0] - sizes[nid][0] / 2.0 for nid in pos]
    xs1 = [pos[nid][0] + sizes[nid][0] / 2.0 for nid in pos]
    ys0 = [pos[nid][1] - sizes[nid][1] / 2.0 for nid in pos]
    ys1 = [pos[nid][1] + sizes[nid][1] / 2.0 for nid in pos]
    minx, maxx = min(xs0), max(xs1)
    miny, maxy = min(ys0), max(ys1)
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    return minx - margin * dx, maxx + margin * dx, miny - margin * dy, maxy + margin * dy


def initial_fontsize_for_ax(ax: plt.Axes) -> float:
    """Pick a pt size so that a line of text has bbox-height equal to
    ``CHAR_H_DATA`` in the axes' *actual* data-to-pixel transform.
    We measure the live ``transData`` after ``apply_aspect()`` so the value is
    correct even when matplotlib adjusts the axes to match aspect='equal'."""
    ax.apply_aspect()
    fig = ax.figure
    t0 = ax.transData.transform((0.0, 0.0))
    t1 = ax.transData.transform((0.0, 1.0))
    pixels_per_data_y = max(abs(t1[1] - t0[1]), 1e-6)
    target_bbox_px = CHAR_H_DATA * pixels_per_data_y
    # Empirically (measured on this mpl build) text bbox height in pixels
    # equals fontsize * dpi / 72 within ~3%, so fontsize_pt = bbox_px*72/dpi.
    dpi = fig.dpi
    fs_pt = target_bbox_px * 72.0 / dpi
    # 0.88 safety factor leaves a small vertical margin inside CHAR_H_DATA so
    # descenders never touch the box edge. No hard minimum: when fs is too
    # small to render cleanly we hide the label entirely (see MIN_READABLE_FS).
    return max(0.01, 0.88 * fs_pt)


# Below this fontsize (in pt) matplotlib's pixel rounding makes character
# width unpredictable, so we hide labels rather than let them overflow their
# boxes. Boxes themselves stay visible; zooming in restores the label.
MIN_READABLE_FS = 4.0


def sympy_display_str(expr: sp.Expr, use_latex: bool) -> str:
    """symengine ``log`` is the natural logarithm (base e), same as ln."""
    if use_latex:
        try:
            return f"${sp.latex(expr)}$".replace(r"\log", r"\ln")
        except Exception:
            return str(expr)
    return str(expr).replace("log(", "ln(")


def draw_networkx_edges_curved(
    ax: plt.Axes, g: nx.Graph, pos: Dict[int, tuple]
) -> None:
    """Draw each edge with its own arc so lines do not sit on top of each other."""
    for u, v in g.edges():
        lo, hi = (u, v) if u < v else (v, u)
        h = hash((lo, hi)) % 10000
        rad = 0.06 + (h % 180) / 900.0
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            arrows=True,
            arrowstyle="-",
            alpha=EDGE_ALPHA,
            edge_color=EDGE_COLOR,
            width=EDGE_WIDTH,
        )
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            p.set_zorder(2)


def set_limits_from_positions(ax: plt.Axes, pos: Dict[int, tuple], margin: float) -> None:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = max(maxx - minx, 1e-6)
    dy = max(maxy - miny, 1e-6)
    ax.set_xlim(minx - margin * dx, maxx + margin * dx)
    ax.set_ylim(miny - margin * dy, maxy + margin * dy)


def _apply_zoom(ax: plt.Axes, scale: float, cx: float, cy: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(cx - (cx - x0) / scale, cx + (x1 - cx) / scale)
    ax.set_ylim(cy - (cy - y0) / scale, cy + (y1 - cy) / scale)


def connect_zoom_interactions(fig: plt.Figure, ax: plt.Axes) -> Tuple[int, int]:
    def on_scroll(event) -> None:
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if getattr(event, "step", 0) != 0:
            scale = 1.15 if event.step > 0 else 1.0 / 1.15
        elif event.button == "up":
            scale = 1.15
        else:
            scale = 1.0 / 1.15
        _apply_zoom(ax, scale, event.xdata, event.ydata)
        update_zoom_dependent_style(ax)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("+", "=", "plus"):
            scale = 1.15
        elif event.key in ("-", "_", "minus"):
            scale = 1.0 / 1.15
        else:
            return
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        _apply_zoom(ax, scale, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
        update_zoom_dependent_style(ax)
        fig.canvas.draw_idle()

    cid1 = fig.canvas.mpl_connect("scroll_event", on_scroll)
    cid2 = fig.canvas.mpl_connect("key_press_event", on_key)
    return cid1, cid2


def _view_size_sqrt(ax: plt.Axes) -> float:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return math.sqrt(max(x1 - x0, 1e-15) * max(y1 - y0, 1e-15))


def update_zoom_dependent_style(ax: plt.Axes) -> None:
    """Node boxes live in data coordinates, so they scale by themselves when you
    zoom. Only text font size and edge stroke width are pt-based and therefore
    need to track the view so the picture stays in proportion."""
    ref = getattr(ax, "_eml_ref_span", None)
    base_fs = getattr(ax, "_eml_base_fs", None)
    base_lw = getattr(ax, "_eml_base_edge_lw", None)
    if ref is None or base_fs is None or base_lw is None:
        return
    ratio = ref / max(_view_size_sqrt(ax), 1e-15)
    fs = max(0.01, base_fs * ratio)
    lw = max(0.15, min(base_lw * ratio, 12.0))
    show_labels = fs >= MIN_READABLE_FS
    for t in ax.texts:
        if not getattr(t, "_eml_node_label", False):
            continue
        t.set_fontsize(fs)
        t.set_visible(show_labels)
    for p in ax.patches:
        if isinstance(p, FancyArrowPatch):
            p.set_linewidth(lw)
        elif isinstance(p, FancyBboxPatch) and getattr(p, "_eml_node_patch", False):
            p.set_linewidth(max(0.25, min(base_lw * ratio * 1.2, 5.0)))
    for col in ax.collections:
        if isinstance(col, LineCollection):
            col.set_linewidth(lw)


def connect_zoom_font_sync(fig: plt.Figure, ax: plt.Axes) -> Tuple[int, int]:
    def on_lim(ax2: plt.Axes) -> None:
        if ax2 is not ax:
            return
        if getattr(ax, "_eml_rendering", False):
            return
        update_zoom_dependent_style(ax)
        fig.canvas.draw_idle()

    c1 = ax.callbacks.connect("xlim_changed", on_lim)
    c2 = ax.callbacks.connect("ylim_changed", on_lim)
    return c1, c2


# ----------------------------- render ------------------------------


def label_for(value: sp.Expr, use_latex: bool) -> str:
    return sympy_display_str(value, use_latex)


# Labels longer than this get truncated in the rendered node box; the full
# symbolic expression is available via the on-click inspector.
MAX_LABEL_CHARS = 24


def truncate_label(s: str) -> str:
    if len(s) <= MAX_LABEL_CHARS:
        return s
    return s[: MAX_LABEL_CHARS - 1] + "\u2026"


def node_face_edge_colors(value: sp.Basic) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Per-node (face, edge) RGBA. Recognised constants get a saturated face
    + matching solid border so they pop; everything else uses the muted
    default palette."""
    highlight = constant_face_color(value)
    if highlight is not None:
        return (
            mcolors.to_rgba(highlight, alpha=0.85),
            mcolors.to_rgba(highlight, alpha=1.0),
        )
    return (
        mcolors.to_rgba(NODE_FACE, alpha=NODE_FACE_ALPHA),
        mcolors.to_rgba(NODE_EDGE, alpha=0.92),
    )


def draw_node_boxes_and_labels(
    ax: plt.Axes,
    pos: Dict[int, List[float]],
    sizes: Dict[int, Tuple[float, float]],
    labels: Dict[int, str],
    values: Dict[int, sp.Basic],
    font_size: float,
) -> None:
    for nid, (x, y) in pos.items():
        w, h = sizes[nid]
        face, edge = node_face_edge_colors(values[nid])
        patch = FancyBboxPatch(
            (x - w / 2.0, y - h / 2.0),
            w, h,
            boxstyle=f"round,pad=0,rounding_size={min(w, h) * 0.22}",
            fc=face, ec=edge, linewidth=1.1, zorder=4,
        )
        patch._eml_node_patch = True
        patch._eml_node_id = nid
        ax.add_patch(patch)
        t = ax.text(
            x, y, labels[nid],
            fontsize=font_size, ha="center", va="center", color=NODE_TEXT,
            zorder=5,
        )
        t._eml_node_label = True
        t._eml_node_pos = (x, y)
        # Clip the label to the axes rectangle (pixel-accurate), so when the
        # node box is cut off at the edge the label is cut off the same way
        # instead of bleeding into the gutter as a "detached" string.
        t.set_clip_on(True)
        t.set_clip_box(ax.bbox)
        if font_size < MIN_READABLE_FS:
            t.set_visible(False)


def constants_present_in(values: Dict[int, sp.Basic]) -> List[sp.Basic]:
    """The subset of recognised constants that actually appear in this state,
    in the canonical CONSTANT_FACES order. Empty list -> no legend needed."""
    seen = set(values.values())
    return [c for c in CONSTANT_FACES if c in seen]


def create_constant_legend(ax: plt.Axes, values: Dict[int, sp.Basic]) -> None:
    """Top-right legend: one bold colored symbol per recognised constant that
    actually appears in the current state. Each symbol is drawn in the same
    highlight color as the matching node boxes."""
    present = constants_present_in(values)
    if not present:
        return
    x = 0.99
    for c in reversed(present):
        ax.text(
            x, 0.985, CONSTANT_LABELS[c],
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=13, fontweight="bold",
            color=CONSTANT_FACES[c],
            zorder=20,
        )
        x -= 0.035


def create_full_label_inspector(ax: plt.Axes) -> None:
    """One fixed text overlay (axes coords) that shows the full symbolic value
    of whichever node the user last clicked. Recreated each render."""
    ann = ax.text(
        0.01, 0.985, "click a node to show its full value",
        transform=ax.transAxes,
        ha="left", va="top",
        color=NODE_TEXT, fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc=mcolors.to_rgba(NODE_FACE, alpha=0.85),
            ec=NODE_EDGE, linewidth=0.9,
        ),
        zorder=20,
        wrap=True,
    )
    ax._eml_full_label_annotation = ann


def _hit_node_patch(ax: plt.Axes, x: float, y: float):
    for p in ax.patches:
        if not getattr(p, "_eml_node_patch", False):
            continue
        px, py = p.get_x(), p.get_y()
        if px <= x <= px + p.get_width() and py <= y <= py + p.get_height():
            return p
    return None


def connect_node_click_inspector(fig: plt.Figure, ax: plt.Axes) -> int:
    def on_click(event) -> None:
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        ann = getattr(ax, "_eml_full_label_annotation", None)
        if ann is None:
            return
        hit = _hit_node_patch(ax, event.xdata, event.ydata)
        if hit is None:
            return
        nid = getattr(hit, "_eml_node_id", None)
        full = getattr(ax, "_eml_full_labels", {}).get(nid, "")
        ann.set_text(f"[{nid}]   {full}")
        fig.canvas.draw_idle()
    return fig.canvas.mpl_connect("button_press_event", on_click)


def format_dim_segment(label: str, dim: Optional[float]) -> Optional[str]:
    """One '<label> ≈ <value>' pill, or None when the estimate is unavailable."""
    if dim is None:
        return None
    return f"{label} ≈ {dim:.2f}"


def format_title(step_idx: int, n_nodes: int, n_links: int, g: nx.Graph) -> str:
    """Header above the graph. We show two flavours of Wolfram-physics Δ:
      * dim_avg    -- averaged over every node as source (global view)
      * dim_center -- measured from a graph-center node (an observer sitting
                      at the most 'inside' point of the universe)
    Pure tree topology (|E| == |V| - 1) is flagged because Δ for a tree
    cannot settle at a finite non-integer d -- it just keeps climbing with
    the diameter."""
    base = f"Step {step_idx}   |   {n_nodes} nodes,  {n_links} links"
    segments = [base]
    for label, value in (
        ("dim_avg", compute_dimension(g)),
        ("dim_center", compute_dimension_from_center(g)),
    ):
        seg = format_dim_segment(label, value)
        if seg is not None:
            segments.append(seg)
    if len(segments) > 1 and g.number_of_edges() == g.number_of_nodes() - 1:
        segments[-1] = segments[-1] + " (acyclic)"
    return "   |   ".join(segments)


def render(ax: plt.Axes, state: GraphState, step_idx: int, use_latex: bool = False) -> None:
    fig = ax.figure
    ax._eml_rendering = True
    try:
        ax.clear()
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        # "datalim" keeps the axes rectangle fixed and expands the data limits
        # to match aspect='equal'. That ensures every node position stays
        # inside the axes rect (so boxes/links aren't orphaned into the gutter
        # between axes and figure). initial_fontsize_for_ax calls
        # ax.apply_aspect() before reading transData so the fontsize accounts
        # for the post-adjustment limits.
        ax.set_aspect("equal", adjustable="datalim")

        g = build_nx_graph(state)
        n = len(state.values)
        full_labels = {nid: label_for(v, use_latex) for nid, v in state.values.items()}
        labels = {nid: truncate_label(s) for nid, s in full_labels.items()}
        ax._eml_full_labels = full_labels
        aspects = measure_label_aspects(list(labels.values()), fig)
        sizes = {
            nid: estimate_box_size_data(labels[nid], aspects[labels[nid]])
            for nid in labels
        }

        pos_dict = compute_layout(g, state)
        pos = {nid: [float(p[0]), float(p[1])] for nid, p in pos_dict.items()}
        pos = scale_layout_for_boxes(pos, sizes)
        pos = pack_rectangles_no_overlap(pos, sizes)

        minx, maxx, miny, maxy = compute_view_bounds(pos, sizes, margin=0.18)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        draw_networkx_edges_curved(ax, g, {nid: tuple(p) for nid, p in pos.items()})
        fs = initial_fontsize_for_ax(ax)
        draw_node_boxes_and_labels(ax, pos, sizes, labels, state.values, fs)

        ax.set_title(
            format_title(step_idx, n, len(state.edges), g),
            color=TITLE_COLOR,
        )
        ax.set_axis_off()
        create_full_label_inspector(ax)
        create_constant_legend(ax, state.values)

        ax._eml_ref_span = _view_size_sqrt(ax)
        ax._eml_base_fs = float(fs)
        ax._eml_base_edge_lw = float(EDGE_WIDTH)
        update_zoom_dependent_style(ax)
    finally:
        ax._eml_rendering = False


# ------------------------------- ui --------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive EML universe graph viewer.")
    p.add_argument(
        "--latex",
        action="store_true",
        help="use LaTeX/mathtext labels (slow for many nodes); default is SymPy sstr()",
    )
    u = p.add_mutually_exclusive_group()
    u.add_argument(
        "--random",
        action="store_true",
        help="use random EML direction per link (skip interactive prompt)",
    )
    u.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic universe (skip interactive prompt)",
    )
    u.add_argument(
        "--merge",
        action="store_true",
        help="use value-merging universe (same-value nodes collapse)",
    )
    return p.parse_args()


def resolve_graph_universe(args: argparse.Namespace) -> str:
    if args.random:
        return "random"
    if args.deterministic:
        return "deterministic"
    if args.merge:
        return "merge"
    print("EML graph viewer")
    while True:
        s = input(
            "Universe: [d]eterministic, [r]andom, or [m]erge (default d): "
        ).strip().lower()
        if s in ("", "d", "det", "deterministic"):
            return "deterministic"
        if s in ("r", "rand", "random"):
            return "random"
        if s in ("m", "merge"):
            return "merge"
        print("  Type d, r, or m.")


def main() -> None:
    args = parse_args()
    universe = resolve_graph_universe(args)
    if universe == "random":
        from eml_universe_random import initial_state, step
    elif universe == "merge":
        from eml_universe_merge import initial_state, step
    else:
        from eml_universe import initial_state, step

    use_latex = args.latex
    history = [initial_state()]
    cursor = [0]

    fig, ax = plt.subplots(figsize=(13, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    plt.subplots_adjust(bottom=0.12)

    def redraw() -> None:
        render(ax, history[cursor[0]], cursor[0], use_latex=use_latex)
        fig.canvas.draw_idle()

    def on_next(_event) -> None:
        if cursor[0] + 1 >= len(history):
            history.append(step(history[cursor[0]]))
        cursor[0] += 1
        redraw()

    def on_back(_event) -> None:
        if cursor[0] > 0:
            cursor[0] -= 1
            redraw()

    ax_back = plt.axes([0.70, 0.02, 0.10, 0.05], facecolor=BTN_FACE)
    ax_next = plt.axes([0.82, 0.02, 0.10, 0.05], facecolor=BTN_FACE)
    style_nav_button_axes(ax_back)
    style_nav_button_axes(ax_next)
    btn_back = Button(ax_back, "← Back", color=BTN_FACE, hovercolor=BTN_HOVER)
    btn_next = Button(ax_next, "Next →", color=BTN_FACE, hovercolor=BTN_HOVER)
    btn_back.label.set_color(BTN_LABEL)
    btn_next.label.set_color(BTN_LABEL)
    for btn in (btn_back, btn_next):
        outline_matplotlib_button(btn)
    btn_back.on_clicked(on_back)
    btn_next.on_clicked(on_next)
    fig._eml_buttons = (btn_back, btn_next)  # keep refs alive

    style_matplotlib_toolbar(fig)
    connect_zoom_interactions(fig, ax)
    connect_zoom_font_sync(fig, ax)
    connect_node_click_inspector(fig, ax)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
