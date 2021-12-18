from __future__ import annotations

import argparse
import os.path
from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Callable
from typing import Union

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


@dataclass
class Node:
    left: Node | Leaf | None
    right: Node | Leaf | None
    height: int
    parent: Node | None

    def __str__(self):
        def _child_to_str(child: Node | Leaf | None) -> str:
            if child is None:
                return "None"
            elif isinstance(child, Leaf):
                return str(child.value)
            elif isinstance(child, Node):
                return "Node(...)"
            else:
                # should raise here
                pass

        return (
            f"Node(left={_child_to_str(self.left)}, "
            f"right={_child_to_str(self.right)}, height={self.height})"
        )


@dataclass
class Leaf:
    parent: Node
    value: int

    def __int__(self):
        return self.value

    def __str__(self):
        return f"Leaf({self.value})"


TNode = Union[Node, Leaf]


def is_tree(n: TNode) -> bool:
    return isinstance(n, Node)


def traverse(node: Node, func: Callable[[Node], None]) -> None:
    def traverse_rec(n: Node) -> None:
        func(n)
        if isinstance(n.left, Node):
            traverse_rec(n.left)
        if isinstance(n.right, Node):
            traverse_rec(n.right)

    traverse_rec(node)


def lazy_copy(node: Node) -> Node:
    expr = node_to_expression(node)
    return parse(expr)


def add_snail_numbers(lhs: Node, rhs: Node) -> Node:
    new_left = lazy_copy(lhs)
    new_right = lazy_copy(rhs)

    def increase_height(n: Node) -> None:
        n.height += 1

    traverse(new_left, increase_height)
    traverse(new_right, increase_height)

    new_parent = Node(parent=None, left=new_left, right=new_right, height=0)

    return new_parent


def parse(expression: str, ignore_whitespace: bool = True) -> TNode:
    """
    Parses a string expression into an AST, each node has two children, each child can
    be either another node or a terminal node with a value.

    Example:
        expression: [[1,2],3], will parse to the following tree

                *
               / \
              *   3
            /  \
           1   2
    """
    stack: list[Node] = []

    height = 0
    last_char: str | None = None
    last_leaf: Leaf | None = None  # Uglies hack for 2 digit numbers

    i: int
    c: str

    root: Node | None = None

    for i, c in enumerate(expression):
        if c.isspace() and ignore_whitespace:
            continue
        if last_char is None:  # special case of the top node
            if c != "[":
                raise ValueError(f"Expected '[' char as the first token: {i}")
            root = Node(left=None, right=None, height=height, parent=None)
            stack.append(root)
        else:
            if c == "[":
                if not stack:
                    raise ValueError(f"Expected parent node to exist: {i}")
                parent = stack[-1]  # peek at stack
                height += 1
                node = Node(left=None, right=None, height=height, parent=parent)
                # an open "[" can appear after , or another [ this determines what
                # side of parent it belongs to
                if last_char == "[":
                    parent.left = node
                elif last_char.isdigit() or last_char == ",":
                    parent.right = node
                else:
                    raise ValueError(
                        f"Unexpected character: {c} expected to follow ',', '[' or a "
                        f"digit: {i}"
                    )
                stack.append(node)
            elif c == ",":
                if not last_char:
                    raise ValueError(f"Parse error, previous token is empty: {i}")
                if last_char != "]" and not last_char.isdigit():
                    raise ValueError(
                        f"Parse error, expected previous token to be a digit or "
                        f"closing bracket ']' : {i}"
                    )
            elif c.isdigit():
                if last_leaf is not None and last_char.isdigit():
                    tmp = last_leaf.value
                    last_leaf.value = tmp * 10 + int(c)
                    last_leaf = None
                    continue
                if not stack:
                    raise ValueError(f"Expected parent node to exist: {i}")
                parent = stack[-1]
                leaf = Leaf(parent=parent, value=int(c))
                last_leaf = leaf
                if last_char == "[":
                    parent.left = leaf
                elif last_char == ",":
                    parent.right = leaf

                else:
                    raise ValueError(f"Expected digit to occur after")
            elif c == "]":
                stack.pop()
                height -= 1
            else:
                raise ValueError(f"Unexpected value of: {c} at {i}")
        last_char = c

    if stack:
        error_msg = ",".join(str(node) for node in stack)
        raise ValueError(f"Unclosed nodes found: [{error_msg}]")
    if not root:
        raise ValueError("Could not parse, fatal error")
    return root


def node_to_expression(node: Node) -> str:
    """
    Renders a tree to a string (inverse of parse)
    """
    result = ""

    def inorder(n: Node):
        nonlocal result
        result += "["
        if is_tree(n.left):
            inorder(n.left)
        else:
            result += f"{n.left.value}"
        result += ","
        if is_tree(n.right):
            inorder(n.right)
        else:
            result += f"{n.right.value}"

        result += "]"

    inorder(node)

    return result


class Side(Enum):
    Left = auto()
    Right = auto()
    Center = auto()


def inorder_traverse(node: Node) -> list[tuple[TNode, Side]]:
    result = []

    def inorder_rec(n: Node, s: Side):
        if isinstance(n, Node):
            inorder_rec(n.left, Side.Left)

        result.append((n, s))
        if isinstance(n, Node):
            inorder_rec(n.right, Side.Right)

    inorder_rec(node, Side.Center)

    return result


def explode(node: Node) -> bool:
    # find the first node with height == 4 and has pairs

    def should_explode(item: tuple[Node, Side]):
        n, s = item
        return (
            isinstance(n, Node)
            and isinstance(n.left, Leaf)
            and isinstance(n.right, Leaf)
            and n.height >= 4
            and s != Side.Center
        )

    # Find node we need to explode if any
    all_nodes: list[tuple[TNode, Side]] = inorder_traverse(node)
    node_to_explode_side = Side.Center
    node_to_explode: Node | None = None
    pos = -1

    for idx, (nd, sd) in enumerate(all_nodes):
        if should_explode((nd, sd)):
            # We found the node we need to explode
            node_to_explode = nd
            pos = idx
            node_to_explode_side = sd
            break

    if not node_to_explode:
        return False

    # move back from the index to find first Leaf and add value, we have to move
    # one more to the right, and one more to the left since the immediate left/right
    # are the nodes we are exploding into one
    for idx in reversed(range(0, pos - 1)):
        tmp_node, _ = all_nodes[idx]
        if isinstance(tmp_node, Leaf):
            tmp_node.value += node_to_explode.left.value
            break

    for idx in range(pos + 2, len(all_nodes)):
        tmp_node, _ = all_nodes[idx]
        if isinstance(tmp_node, Leaf):
            tmp_node.value += node_to_explode.right.value
            break

    # Explode the node
    parent = node_to_explode.parent
    zero_leaf = Leaf(value=0, parent=parent)
    if node_to_explode_side == Side.Left:
        parent.left = zero_leaf
    else:
        parent.right = zero_leaf

    return True


def split(node: Node) -> bool:
    all_nodes: list[tuple[TNode, Side]] = inorder_traverse(node)

    child_side = Side.Center
    node_to_split: Leaf | None = None

    for idx, (nd, sd) in enumerate(all_nodes):
        if isinstance(nd, Leaf):
            if nd.value > 9:
                node_to_split = nd
                child_side = sd
                break

    if node_to_split is None:
        return False

    parent = node_to_split.parent

    new_node = Node(height=parent.height + 1, parent=parent, left=None, right=None)
    leaf_left = Leaf(value=node_to_split.value // 2, parent=new_node)
    leaf_right = Leaf(
        value=node_to_split.value // 2 + node_to_split.value % 2, parent=new_node
    )
    new_node.left = leaf_left
    new_node.right = leaf_right

    if child_side == Side.Left:
        parent.left = new_node
    else:
        parent.right = new_node

    return True


def get_magnitude(node: Node) -> int:
    def get_value_rec(n: Node) -> int:
        if isinstance(n.left, Leaf):
            left_value = n.left.value
        else:
            left_value = get_value_rec(n.left)

        if isinstance(n.right, Leaf):
            right_value = n.right.value
        else:
            right_value = get_value_rec(n.right)

        return 3 * left_value + 2 * right_value

    return get_value_rec(node)


def reduce_snailfish(node: Node):
    while explode(node) or split(node):
        pass


def part_1(s: str) -> int:
    lines = s.splitlines()

    snail_number = parse(lines[0])

    for line in lines[1:]:
        next_snail_number = parse(line)
        snail_number = add_snail_numbers(snail_number, next_snail_number)
        reduce_snailfish(snail_number)

    return get_magnitude(snail_number)


def part_2(s: str) -> int:
    max_result = 0
    lines = s.splitlines()

    for idx1 in range(len(lines) - 1):
        for idx2 in range(idx1 + 1, len(lines)):
            a_node = parse(lines[idx1])
            b_node = parse(lines[idx2])
            add_1 = add_snail_numbers(a_node, b_node)
            reduce_snailfish(add_1)

            add_2 = add_snail_numbers(b_node, a_node)
            reduce_snailfish(add_2)

            max_result = max(max_result, get_magnitude(add_1), get_magnitude(add_2))

    return max_result


def compute(s: str, part: int) -> int:
    if part == 1:
        return part_1(s)
    else:
        return part_2(s)


INPUT_S = """\
[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
[[[5,[2,8]],4],[5,[[9,9],0]]]
[6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
[[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
[[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
[[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
[[[[5,4],[7,7]],8],[[8,3],8]]
[[9,3],[[9,9],[6,[4,9]]]]
[[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
[[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]
"""


@pytest.mark.parametrize(
    ("input_s", "part", "expected"),
    (
        (INPUT_S, 1, 4140),
        (INPUT_S, 2, 3993),
    ),
)
def test(input_s: str, part: int, expected: int) -> None:
    assert compute(input_s, part) == expected


@pytest.mark.parametrize(
    ("input_s", "expected"),
    (
        ("[[[[[9,8],1],2],3],4]", "[[[[0,9],2],3],4]"),
        ("[7,[6,[5,[4,[3,2]]]]]", "[7,[6,[5,[7,0]]]]"),
        ("[[6,[5,[4,[3,2]]]],1]", "[[6,[5,[7,0]]],3]"),
        ("[[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]", "[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]"),
        ("[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]", "[[3,[2,[8,0]]],[9,[5,[7,0]]]]"),
    ),
)
def test_explosion(input_s: str, expected: str) -> None:
    tree = parse(input_s)
    explode(tree)
    exploded = node_to_expression(tree)

    assert exploded == expected


@pytest.mark.parametrize(
    ("input_s", "expected"),
    (
        ("[[[[0,7],4],[15,[0,13]]],[1,1]]", "[[[[0,7],4],[[7,8],[0,13]]],[1,1]]"),
        ("[[[[0,7],4],[[7,8],[0,13]]],[1,1]]", "[[[[0,7],4],[[7,8],[0,[6,7]]]],[1,1]]"),
    ),
)
def test_split(input_s: str, expected: str):
    tree = parse(input_s)
    split(tree)
    splitted = node_to_expression(tree)

    assert splitted == expected


@pytest.mark.parametrize(
    ("input_s", "expected"),
    (
        ("[[1,2],[[3,4],5]]", 143),
        ("[[[[0,7],4],[[7,8],[6,0]]],[8,1]]", 1384),
        ("[[[[1,1],[2,2]],[3,3]],[4,4]]", 445),
        ("[[[[3,0],[5,3]],[4,4]],[5,5]]", 791),
        ("[[[[5,0],[7,4]],[5,5]],[6,6]]", 1137),
        ("[[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]", 3488),
    ),
)
def test_get_magnitude(input_s: str, expected: int) -> None:
    tree = parse(input_s)
    actual = get_magnitude(tree)

    assert actual == expected


@pytest.mark.parametrize(
    "expression",
    (
        "[1,2]",
        "[[1,2],3]",
        "[9,[8,7]]",
        "[[1,9],[8,5]]",
        "[[[[1,3],[5,3]],[[1,3],[8,7]]],[[[4,9],[6,9]],[[8,2],[7,3]]]]",
    ),
)
def test_node_to_expression(expression: str) -> None:
    tree = parse(expression)
    actual = node_to_expression(tree)

    assert actual == expression


def test_add() -> None:
    lhs = parse("[1,2]")
    rhs = parse("[[3,4],5]")

    result = node_to_expression(add_snail_numbers(lhs, rhs))

    assert result == "[[1,2],[[3,4],5]]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--p", type=int, default=1)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
