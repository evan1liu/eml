"""
EML rewrite universe with **value-driven node identification**.

This is the variant suggested in chat: extend the deterministic universe with
a third phase that lets the algebra drive topology. After Phase 1 (link
subdivision) and Phase 2 (`1`-leaf expansion), every group of nodes that
carry the same SymEngine value is merged into a single node, with the union
of their incident edges (minus self-loops) becoming the new node's
neighbourhood.

Because Phase 2 spawns many literal `1`s every step, the most aggressive
policy (chosen by user) merges those too — they all collapse into one shared
"1 hub" which is then expanded to `e` by the next step's Phase 2. That hub,
together with deeper algebraic coincidences (two distinct EML chains
simplifying to the same expression), is what creates cycles and breaks the
tree topology of the original `eml_universe`.
"""

import copy
from typing import Dict, List

import symengine as sp
from symengine import Integer

from eml_universe import (
    GraphState,
    consume_links_phase,
    expand_ones_phase,
    initial_state,
)


# `1` is the bedrock of this universe -- Phase 2 keeps producing fresh 1-leaves
# as the universe's "unused material", so collapsing them all into a single
# hub would destroy the spatial meaning of the 1-boundary. Every OTHER
# algebraic coincidence still triggers a merge.
MERGE_EXCLUDED_VALUES = frozenset({Integer(1)})


def group_node_ids_by_value(state: GraphState) -> Dict[sp.Basic, List[int]]:
    groups: Dict[sp.Basic, List[int]] = {}
    for nid, val in state.values.items():
        groups.setdefault(val, []).append(nid)
    return groups


def build_rename_table(state: GraphState) -> Dict[int, int]:
    """Map every node id that should disappear to the surviving id of its
    equal-value class (the smallest id wins). Classes whose value is in
    MERGE_EXCLUDED_VALUES are left untouched."""
    rename: Dict[int, int] = {}
    for value, ids in group_node_ids_by_value(state).items():
        if len(ids) <= 1:
            continue
        if value in MERGE_EXCLUDED_VALUES:
            continue
        keep = min(ids)
        for old in ids:
            if old != keep:
                rename[old] = keep
    return rename


def apply_rename_to_edges(state: GraphState, rename: Dict[int, int]) -> None:
    """Replace renamed endpoints in every edge; self-loops are discarded
    because an undirected graph has no edge from a node to itself."""
    new_edges = set()
    for edge in state.edges:
        a, b = tuple(edge)
        ra = rename.get(a, a)
        rb = rename.get(b, b)
        if ra != rb:
            new_edges.add(frozenset({ra, rb}))
    state.edges = new_edges


def drop_renamed_values(state: GraphState, rename: Dict[int, int]) -> None:
    for old in rename:
        state.values.pop(old, None)


def merge_equal_values_phase(state: GraphState) -> None:
    rename = build_rename_table(state)
    if not rename:
        return
    apply_rename_to_edges(state, rename)
    drop_renamed_values(state, rename)


def step(state: GraphState) -> GraphState:
    new_state = copy.deepcopy(state)
    consume_links_phase(state, new_state)
    expand_ones_phase(state, new_state)
    merge_equal_values_phase(new_state)
    return new_state


def evolve(steps: int) -> List[GraphState]:
    history = [initial_state()]
    for _ in range(steps):
        history.append(step(history[-1]))
    return history
