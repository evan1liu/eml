"""
Matplotlib-based visual display for the EML rewrite universe.

Loads the rewrite rules from `eml_universe` and renders each step in an
interactive matplotlib window with `← Back` and `Next →` buttons.

By default node labels use SymPy's plain string form (fast). Pass --latex to
use matplotlib mathtext + LaTeX from sympy (slow on large graphs).

Run:
    source venv/bin/activate && python3 eml_graph.py
    source venv/bin/activate && python3 eml_graph.py --latex --random

You will be asked **[d]eterministic** vs **[r]andom** unless you pass
``--deterministic`` or ``--random``.

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
from typing import Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import symengine as sp
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.widgets import Button

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
LABEL_REF_FS_PT = 100.0  # font size used only for measuring aspect ratios


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
    """Pt size so one line of text (CHAR_H_DATA data tall) maps to ~ that many
    data-to-point units at the axes' current transform."""
    y0, y1 = ax.get_ylim()
    bbox = ax.get_position()
    fig = ax.figure
    fig_h_inches = fig.get_size_inches()[1]
    ax_h_inches = max(1e-3, bbox.height * fig_h_inches)
    span_y = max(y1 - y0, 1e-6)
    inches_per_data = ax_h_inches / span_y
    pt_for_char_h = 72.0 * inches_per_data * CHAR_H_DATA
    return max(4.0, min(40.0, 0.72 * pt_for_char_h))


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
    fs = max(1.5, base_fs * ratio)
    lw = max(0.15, min(base_lw * ratio, 12.0))
    for t in ax.texts:
        if not getattr(t, "_eml_node_label", False):
            continue
        t.set_fontsize(fs)
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


def draw_node_boxes_and_labels(
    ax: plt.Axes,
    pos: Dict[int, List[float]],
    sizes: Dict[int, Tuple[float, float]],
    labels: Dict[int, str],
    font_size: float,
) -> None:
    face = mcolors.to_rgba(NODE_FACE, alpha=NODE_FACE_ALPHA)
    edge = mcolors.to_rgba(NODE_EDGE, alpha=0.92)
    for nid, (x, y) in pos.items():
        w, h = sizes[nid]
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


def render(ax: plt.Axes, state: GraphState, step_idx: int, use_latex: bool = False) -> None:
    fig = ax.figure
    ax._eml_rendering = True
    try:
        ax.clear()
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
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

        minx, maxx, miny, maxy = compute_view_bounds(pos, sizes, margin=0.08)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        draw_networkx_edges_curved(ax, g, {nid: tuple(p) for nid, p in pos.items()})
        fs = initial_fontsize_for_ax(ax)
        draw_node_boxes_and_labels(ax, pos, sizes, labels, fs)

        ax.set_title(
            f"Step {step_idx}   |   {n} nodes,  {len(state.edges)} links",
            color=TITLE_COLOR,
        )
        ax.set_axis_off()
        create_full_label_inspector(ax)

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
    return p.parse_args()


def resolve_graph_universe(args: argparse.Namespace) -> str:
    if args.random:
        return "random"
    if args.deterministic:
        return "deterministic"
    print("EML graph viewer")
    while True:
        s = input("Universe: [d]eterministic or [r]andom (default d): ").strip().lower()
        if s in ("", "d", "det", "deterministic"):
            return "deterministic"
        if s in ("r", "rand", "random"):
            return "random"
        print("  Type d or r.")


def main() -> None:
    args = parse_args()
    universe = resolve_graph_universe(args)
    if universe == "random":
        from eml_universe_random import initial_state, step
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
