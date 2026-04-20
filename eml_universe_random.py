"""
Random variant of the EML rewrite universe.

Phase 1 matches `eml_universe` topology: each link {a, b} is replaced by a path
a—c—b with c a single new node (edges {lo, c} and {c, hi} after sorting ids).

Only the **value** on c differs: an undirected edge has no preferred order, so
each link uses a fair coin flip for argument order:

    p = 0.5   c = EML(a, b) = exp(a) - log(b)
    p = 0.5   c = EML(b, a) = exp(b) - log(a)

where `a, b` are the two endpoints of the link (tuple order arbitrary).

Phase 2 is unchanged from `eml_universe`.

Each run uses fresh system randomness (no seed).
"""

import copy
import random
from typing import List

from eml_universe import (
    GraphState,
    eml,
    expand_ones_phase,
    initial_state,
)


def consume_links_phase(state: GraphState, new_state: GraphState) -> None:
    for link in list(state.edges):
        a, b = tuple(link)
        va, vb = state.values[a], state.values[b]
        lo, hi = sorted((a, b))
        if random.random() < 0.5:
            c_val = eml(va, vb)
        else:
            c_val = eml(vb, va)
        c_id = new_state.next_id
        new_state.next_id += 1
        new_state.values[c_id] = c_val
        new_state.edges.discard(link)
        new_state.edges.add(frozenset({lo, c_id}))
        new_state.edges.add(frozenset({c_id, hi}))


def step(state: GraphState) -> GraphState:
    new_state = copy.deepcopy(state)
    consume_links_phase(state, new_state)
    expand_ones_phase(state, new_state)
    return new_state


def evolve(steps: int) -> List[GraphState]:
    history = [initial_state()]
    for _ in range(steps):
        history.append(step(history[-1]))
    return history
