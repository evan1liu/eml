import inspect
from collections import namedtuple
from typing import Callable, Union
from sympy import Symbol, Expr, exp as sym_exp, log as sym_log, simplify

E = namedtuple('E', ['x', 'y'])

Node = Union[int, E, Expr]

def print_tree(node: Union[Node, Callable[..., E]], prefix: str = "", is_last: bool = True, is_root: bool = True) -> None:
    check_reduction = is_root
    if is_root and callable(node) and node is not E:
        params = list(inspect.signature(node).parameters)
        print(f"{node.__name__}({', '.join(params)}) =")
        node = node(*[Symbol(p, real=True) for p in params])
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
    if check_reduction:
        if isinstance(node, E):
            print()
            reduced = reduce_tree(node)
            if reduced is node:
                print("(no reduction found, tree is the same)")
                print()
                print()
                print()
            else:
                print("can be reduced to:")
                print_tree(reduced)
        else:
            print()
            print()
            print()

def evaluate(node: Node) -> Union[int, Expr]:
    if isinstance(node, E):
        return sym_exp(evaluate(node.x)) - sym_log(evaluate(node.y))
    return node

def reduce_tree(tree: E) -> Union[Node, E]:
    val = simplify(evaluate(tree))
    if getattr(val, "is_Symbol", False):
        return val
    if val == 1:
        return 1
    return tree

# Starting to define functions and constants
# Important rules to follow:
# Must not use python default math symbols like +, -, *, or /
# Only use the functions derived from the original EML function
# For example: addition(x, y), subtraction(x, y), multiply(x, y), division(x, y), etc.
# Only one single base constant is allowed which is the number 1

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

# Reduction techniques: input x the output is x also.
# I'm gonna figure out what to do with these two functions later
def LRLR(x: Node) -> E:
    return E(1, E(E(1, E(x, 1)),1))
print_tree(LRLR)

def RLRL(x: Node) -> E:
    return E(E(1, E(E(1, x), 1)), 1)
print_tree(RLRL)

# Continue
# This subtraction may only work for inputs where x > 0
# def subtraction(x: Node, y: Node) -> E:
#     return E(ln(x), exp(y))
# print_tree(subtraction)

# This division function works mathematically,
# but only covers input space where x >= 1 and y > 0
# (x = 1 case relies on extended-reals: ln(0) = -inf)
# def division(x: Node, y: Node) -> E:
#     return exp(E(ln(ln(x)), y))
# print_tree(division)