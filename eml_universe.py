"""
Core engine for the EML rewrite universe (Wolfram-physics style).

State is a graph with:
  - nodes, each carrying a sympy symbolic value
  - undirected links between nodes

Initial state: single node with value 1.

Each step applies, in parallel, on a snapshot of the current graph:

  Phase 1 (subdivide links): for every link {a, b}, create **one** new node c
    whose value is EML(lo, hi) with lo < hi the node ids (canonical order for
    an undirected edge). Replace {a, b} by the path a—c—b: edges {lo, c} and
    {c, hi}. The next step will then operate on those new edges, e.g.
    EML(a, c) and EML(c, b) when those are the links present.

  Phase 2 (expand 1s): every node currently equal to 1 is relabeled to e and
    gains two fresh `1` children linked to it. Node identity (id) persists.

This module has no rendering code.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

import symengine as sp
from symengine import Integer, exp as sym_exp, log as sym_log


@dataclass
class GraphState:
    next_id: int
    values: Dict[int, sp.Expr] = field(default_factory=dict)
    edges: Set[FrozenSet[int]] = field(default_factory=set)


def initial_state() -> GraphState:
    return GraphState(next_id=1, values={0: Integer(1)}, edges=set())


def eml(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    # symengine auto-evaluates exp/log identities (exp(0)=1, log(1)=0,
    # log(exp(x))=x, etc.), so an explicit simplify pass is unnecessary.
    return sym_exp(a) - sym_log(b)


def consume_links_phase(state: GraphState, new_state: GraphState) -> None:
    for link in list(state.edges):
        a, b = tuple(link)
        lo, hi = sorted((a, b))
        v_lo, v_hi = state.values[lo], state.values[hi]
        c_val = eml(v_lo, v_hi)
        c_id = new_state.next_id
        new_state.next_id += 1
        new_state.values[c_id] = c_val
        new_state.edges.discard(link)
        new_state.edges.add(frozenset({lo, c_id}))
        new_state.edges.add(frozenset({c_id, hi}))


def expand_ones_phase(state: GraphState, new_state: GraphState) -> None:
    e_val = sym_exp(1)
    ones = [nid for nid, v in state.values.items() if v == 1]
    for nid in ones:
        new_state.values[nid] = e_val
        for _ in range(2):
            child_id = new_state.next_id
            new_state.next_id += 1
            new_state.values[child_id] = Integer(1)
            new_state.edges.add(frozenset({nid, child_id}))


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
