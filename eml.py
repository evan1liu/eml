import inspect
from collections import namedtuple
from typing import Callable, Union
from sympy import Symbol, Expr, exp as sym_exp, log as sym_log, simplify

E = namedtuple('E', ['x', 'y'])

Node = Union[int, E, Expr]

def print_tree(node: Union[Node, Callable[..., E]], prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
    if is_root and callable(node) and node is not E:
        params = list(inspect.signature(node).parameters)
        print(f"{node.__name__}({', '.join(params)}) =")
        node = node(*[Symbol(p) for p in params])
    if is_root:
        branch = ""
        child_prefix = ""
    else:
        branch = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")
    label = "E" if isinstance(node, E) else str(node)
    print(f"{prefix}{branch}{label}")
    if isinstance(node, E):
        print_tree(node.x, child_prefix, is_last=False, is_root=False)
        print_tree(node.y, child_prefix, is_last=True, is_root=False)

def evaluate(node: Node) -> Union[int, Expr]:
    if isinstance(node, E):
        return sym_exp(evaluate(node.x)) - sym_log(evaluate(node.y))
    return node

def exp(x: Node) -> E:
    return E(x, 1)
print_tree(exp)

euler = E(1, 1)
print(f"euler = {simplify(evaluate(euler))}")
print_tree(euler)

def ln(x: Node) -> E:
    return E(1, E(E(1, x), 1))
print_tree(ln)

def zero(x: Node) -> E:
    return E(x, E(exp(x), 1))
print(f"zero = {simplify(evaluate(zero(1)))}")
print_tree(zero)

def negation(x: Node) -> E:
    return E(zero(1), E(E(ln(x), euler), 1))
print_tree(negation)

## Reduction techniques: input x the output is x also.
## I'm gonna figure out what to do with these two functions later
def LRLR(x: Node) -> E:
    return E(1, E(E(1, E(x, 1)),1))
print_tree(LRLR)

def RLRL(x: Node) -> E:
    return E(E(1, E(E(1, x), 1)), 1)
print_tree(RLRL)

## Continue
def ln_neg(x: Node) -> E:
    return E(zero(1), E(E(zero(1), negation(x)), 1))
print_tree(ln_neg_x)

def addition(x: Node, y: Node) -> E:
    E(ln_neg(-x))