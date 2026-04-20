"""
Text-based adjacency-list display for the EML rewrite universe.

Prints the full evolution from step 0 through step N.

Run:
    source venv/bin/activate && python3 eml_text.py 4
    source venv/bin/activate && python3 eml_text.py --random 4

You will be asked **[d]eterministic**, **[r]andom**, or **[m]erge** unless
you pass ``--deterministic``, ``--random`` or ``--merge``.
"""

import argparse
from typing import Dict, List

import symengine as sp

from eml_universe import GraphState


def sorted_neighbors(state: GraphState, nid: int) -> List[int]:
    neighbors = [
        next(iter(link - {nid}))
        for link in state.edges
        if nid in link
    ]
    return sorted(neighbors)


def format_value(value: sp.Expr) -> str:
    """symengine ``log`` is natural log; display as ``ln`` like the graph viewer."""
    return str(value).replace("log(", "ln(")


def format_node_header(nid: int, value: sp.Expr, id_width: int) -> str:
    return f"  [{nid:>{id_width}}] {format_value(value)}"


def format_neighbor_line(
    neighbor_id: int, neighbor_value: sp.Expr, id_width: int
) -> str:
    return f"      -> [{neighbor_id:>{id_width}}] {format_value(neighbor_value)}"


def id_column_width(state: GraphState) -> int:
    if not state.values:
        return 1
    return len(str(max(state.values.keys())))


def format_step_header(step_idx: int, state: GraphState) -> str:
    title = f"Step {step_idx}   |   {len(state.values)} nodes,  {len(state.edges)} links"
    return title + "\n" + "-" * len(title)


def format_state(state: GraphState, step_idx: int) -> str:
    lines: List[str] = [format_step_header(step_idx, state)]
    width = id_column_width(state)
    for nid in sorted(state.values.keys()):
        lines.append(format_node_header(nid, state.values[nid], width))
        neighbors = sorted_neighbors(state, nid)
        if not neighbors:
            lines.append("      (isolated)")
            continue
        for neighbor_id in neighbors:
            lines.append(
                format_neighbor_line(neighbor_id, state.values[neighbor_id], width)
            )
    return "\n".join(lines)


def print_evolution(max_step: int, universe: str) -> None:
    if universe == "random":
        from eml_universe_random import evolve
    elif universe == "merge":
        from eml_universe_merge import evolve
    else:
        from eml_universe import evolve
    history = evolve(max_step)
    for step_idx, state in enumerate(history):
        print(format_state(state, step_idx))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the full EML-universe evolution from step 0 through step N."
    )
    parser.add_argument(
        "step", type=int, nargs="?", default=3,
        help="final step index to print (default: 3)",
    )
    u = parser.add_mutually_exclusive_group()
    u.add_argument(
        "--random",
        action="store_true",
        help="random EML direction per link (skip prompt)",
    )
    u.add_argument(
        "--deterministic",
        action="store_true",
        help="deterministic universe (skip prompt)",
    )
    u.add_argument(
        "--merge",
        action="store_true",
        help="value-merging universe (skip prompt)",
    )
    return parser.parse_args()


def resolve_text_universe(args: argparse.Namespace) -> str:
    if args.random:
        return "random"
    if args.deterministic:
        return "deterministic"
    if args.merge:
        return "merge"
    print("EML text export")
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
    if args.step < 0:
        raise SystemExit("step must be >= 0")
    universe = resolve_text_universe(args)
    print_evolution(args.step, universe)


if __name__ == "__main__":
    main()
