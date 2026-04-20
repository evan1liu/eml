"""
EML rewrite universe with conserved charge labels (holographic variant).

Every node carries one extra bit of state besides its symbolic value:
a charge label in {+1, -1, 0} (shorthand P, N, 0). Charges propagate
through Phase 1 so two otherwise-identical bulk nodes on opposite sides
of the graph stay distinguishable, and merge (Phase 3) only fuses nodes
that match in BOTH value and charge.

Initial state at step 0 has three nodes: a neutral center `e`, one P-leaf
with value 1, and one N-leaf with value 1. Two edges {e, P1}, {e, N1}.

Phase 1 (subdivide links): same as the base universe, but the new bulk
node inherits a charge computed by `combine_labels_sign` (sign-of-sum).
P + P = P, N + N = N, P + N = 0 (annihilation), 0 acts as identity.

Phase 2 ("replace" flavor): every value-1 leaf promotes to e, KEEPS its
own charge, and spawns exactly ONE new value-1 child with the SAME
charge. Leaf count stays constant forever -- boundary is fixed, bulk
grows.

Phase 3 (label-aware merge): nodes with the same (value, charge) collapse
into one. Value-1 nodes are excluded so the boundary never collapses
into a single hub. Different charges never merge even at matching value.
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Set, Tuple

import symengine as sp
from symengine import Integer, exp as sym_exp

from eml_universe import GraphState, eml


P = 1
N = -1
NEUTRAL = 0


@dataclass
class HoloGraphState(GraphState):
    labels: Dict[int, int] = field(default_factory=dict)


def initial_state() -> HoloGraphState:
    return HoloGraphState(
        next_id=3,
        values={0: sym_exp(1), 1: Integer(1), 2: Integer(1)},
        edges={frozenset({0, 1}), frozenset({0, 2})},
        labels={0: NEUTRAL, 1: P, 2: N},
    )


def combine_labels_sign(la: int, lb: int) -> int:
    """Charge addition with sign collapse: P+N annihilates to neutral,
    same-sign stays, neutral is identity."""
    s = la + lb
    if s > 0:
        return P
    if s < 0:
        return N
    return NEUTRAL


def consume_links_phase_holo(
    state: HoloGraphState,
    new_state: HoloGraphState,
    combiner: Callable[[int, int], int] = combine_labels_sign,
) -> None:
    for link in list(state.edges):
        a, b = tuple(link)
        lo, hi = sorted((a, b))
        c_val = eml(state.values[lo], state.values[hi])
        c_label = combiner(state.labels[lo], state.labels[hi])
        c_id = new_state.next_id
        new_state.next_id += 1
        new_state.values[c_id] = c_val
        new_state.labels[c_id] = c_label
        new_state.edges.discard(link)
        new_state.edges.add(frozenset({lo, c_id}))
        new_state.edges.add(frozenset({c_id, hi}))


def expand_ones_replace_phase(
    state: HoloGraphState, new_state: HoloGraphState
) -> None:
    """Each value-1 leaf becomes e (charge preserved) and spawns exactly one
    fresh value-1 child carrying the same charge. Leaf count stays constant."""
    e_val = sym_exp(1)
    ones = [nid for nid, v in state.values.items() if v == 1]
    for nid in ones:
        parent_label = state.labels[nid]
        new_state.values[nid] = e_val
        child_id = new_state.next_id
        new_state.next_id += 1
        new_state.values[child_id] = Integer(1)
        new_state.labels[child_id] = parent_label
        new_state.edges.add(frozenset({nid, child_id}))


MERGE_EXCLUDED_VALUES = frozenset({Integer(1)})


def group_by_value_and_label(
    state: HoloGraphState,
) -> Dict[Tuple[sp.Basic, int], List[int]]:
    groups: Dict[Tuple[sp.Basic, int], List[int]] = {}
    for nid, val in state.values.items():
        groups.setdefault((val, state.labels[nid]), []).append(nid)
    return groups


def build_rename_table_holo(state: HoloGraphState) -> Dict[int, int]:
    rename: Dict[int, int] = {}
    for (value, _charge), ids in group_by_value_and_label(state).items():
        if len(ids) <= 1:
            continue
        if value in MERGE_EXCLUDED_VALUES:
            continue
        keep = min(ids)
        for old in ids:
            if old != keep:
                rename[old] = keep
    return rename


def apply_rename_to_edges_holo(
    state: HoloGraphState, rename: Dict[int, int]
) -> None:
    new_edges: Set[FrozenSet[int]] = set()
    for edge in state.edges:
        a, b = tuple(edge)
        ra = rename.get(a, a)
        rb = rename.get(b, b)
        if ra != rb:
            new_edges.add(frozenset({ra, rb}))
    state.edges = new_edges


def drop_renamed_nodes(state: HoloGraphState, rename: Dict[int, int]) -> None:
    for old in rename:
        state.values.pop(old, None)
        state.labels.pop(old, None)


def merge_equal_values_phase_holo(state: HoloGraphState) -> None:
    rename = build_rename_table_holo(state)
    if not rename:
        return
    apply_rename_to_edges_holo(state, rename)
    drop_renamed_nodes(state, rename)


def step(state: HoloGraphState) -> HoloGraphState:
    new_state = copy.deepcopy(state)
    consume_links_phase_holo(state, new_state)
    expand_ones_replace_phase(state, new_state)
    merge_equal_values_phase_holo(new_state)
    return new_state


def evolve(steps: int) -> List[HoloGraphState]:
    history = [initial_state()]
    for _ in range(steps):
        history.append(step(history[-1]))
    return history
