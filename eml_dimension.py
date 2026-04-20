"""
Wolfram-physics graph dimension for an EML universe state.

Reference:
    https://www.wolframphysics.org/technical-introduction/limiting-behavior-and-emergent-geometry/the-notion-of-dimension/

For any node X, V_r(X) is the number of nodes reachable in at most r graph
hops from X (the "volume of a ball of radius r in the graph centered at X").
For a d-dimensional cubic grid, V_r(X) ~ r^d as r grows. The emergent
dimension is therefore the log-log slope of V_r vs r in an intermediate range
of r (large enough to escape grid-scale noise, small enough to escape the
finite-graph edge effects).

For a tree, V_r grows exponentially in r (no power law), so this estimator
keeps climbing without converging -- exactly what Wolfram notes as the tree
case.
"""

import math
from collections import deque
from typing import List, Optional

import networkx as nx


def ball_volumes_from(g: nx.Graph, source: int, max_r: int) -> List[int]:
    """Cumulative V_r for a single source: V_r[i] = #nodes at distance <= i."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in g.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    at = [0] * (max_r + 1)
    for d in dist.values():
        if d <= max_r:
            at[d] += 1
    cum = []
    running = 0
    for c in at:
        running += c
        cum.append(running)
    return cum


def average_ball_volumes(g: nx.Graph, max_r: int) -> List[float]:
    """Average cumulative V_r across every node taken as source."""
    nodes = list(g.nodes())
    if not nodes:
        return []
    sums = [0.0] * (max_r + 1)
    for src in nodes:
        for i, c in enumerate(ball_volumes_from(g, src, max_r)):
            sums[i] += c
    n = len(nodes)
    return [s / n for s in sums]


def loglog_slope(volumes: List[float], r_min: int, r_max: int) -> Optional[float]:
    """Linear regression of log(V_r) on log(r), restricted to r_min..r_max."""
    pts = [
        (r, volumes[r])
        for r in range(r_min, r_max + 1)
        if r < len(volumes) and volumes[r] > 0
    ]
    if len(pts) < 2:
        return None
    xs = [math.log(r) for r, _ in pts]
    ys = [math.log(v) for _, v in pts]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def compute_dimension(g: nx.Graph) -> Optional[float]:
    """Wolfram's Δ averaged over every starting node, fit over an intermediate
    radius range. Returns None only when the graph is too small for the
    estimate to be meaningful (we always need at least 2 fit points)."""
    n = g.number_of_nodes()
    if n < 4:
        return None
    if not nx.is_connected(g):
        return None
    diameter = nx.diameter(g)
    if diameter < 3:
        return None
    # Skip r=1 (just the degree, dominated by local noise) and the last hop
    # (finite-graph edge effect). For small-world graphs (diameter ~ log N)
    # diameter // 3 leaves only 1 point, so we widen to diameter - 1.
    r_min = 2
    r_max = max(r_min + 1, diameter - 1)
    vols = average_ball_volumes(g, r_max)
    return loglog_slope(vols, r_min=r_min, r_max=r_max)


def pick_graph_center(g: nx.Graph) -> int:
    """A node minimising eccentricity. networkx may return several centers --
    we just take the smallest id for determinism so the same step always
    reports the same Δ_center."""
    return min(nx.center(g))


def compute_dimension_from_node(g: nx.Graph, source: int) -> Optional[float]:
    """Δ measured from a single source: log-log slope of V_r(source) vs r,
    fit over r in [2, eccentricity(source) - 1]."""
    if g.number_of_nodes() < 4 or not nx.is_connected(g):
        return None
    ecc = nx.eccentricity(g, source)
    if ecc < 3:
        return None
    r_min = 2
    r_max = max(r_min + 1, ecc - 1)
    vols = [float(v) for v in ball_volumes_from(g, source, r_max)]
    return loglog_slope(vols, r_min=r_min, r_max=r_max)


def compute_dimension_from_center(g: nx.Graph) -> Optional[float]:
    """Δ measured from a graph center -- the most 'inside' viewpoint, where
    the ball grows isotropically for as long as possible before hitting the
    finite boundary."""
    if g.number_of_nodes() < 4 or not nx.is_connected(g):
        return None
    if nx.diameter(g) < 3:
        return None
    return compute_dimension_from_node(g, pick_graph_center(g))
