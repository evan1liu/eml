"""
Probabilistic variant of the holographic universe.

Identical to `eml_universe_holo` except for the P+N case in Phase 1:
instead of annihilating to neutral, a mixed-charge edge collapses to
either P or N with equal probability. Physical reading: CP-violation /
measurement-collapse at matter-antimatter interfaces, where no net
neutral is produced but the outcome is non-deterministic.

All other rules (charge conservation for same-sign, neutral as identity,
replace-flavor leaf expansion, label-aware merge) are unchanged.
"""

import copy
import random
from typing import List

from eml_universe_holo import (
    HoloGraphState,
    N,
    NEUTRAL,
    P,
    consume_links_phase_holo,
    expand_ones_replace_phase,
    initial_state,
    merge_equal_values_phase_holo,
)


def combine_labels_collapse(la: int, lb: int) -> int:
    """Same-sign and neutral+x behave like the deterministic rule; the
    P+N mixed case picks P or N uniformly at random (collapse)."""
    if la == NEUTRAL and lb == NEUTRAL:
        return NEUTRAL
    s = la + lb
    if s > 0:
        return P
    if s < 0:
        return N
    return random.choice((P, N))


def step(state: HoloGraphState) -> HoloGraphState:
    new_state = copy.deepcopy(state)
    consume_links_phase_holo(state, new_state, combiner=combine_labels_collapse)
    expand_ones_replace_phase(state, new_state)
    merge_equal_values_phase_holo(new_state)
    return new_state


def evolve(steps: int) -> List[HoloGraphState]:
    history = [initial_state()]
    for _ in range(steps):
        history.append(step(history[-1]))
    return history
