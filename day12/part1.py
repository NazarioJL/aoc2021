from __future__ import annotations

import argparse
import os.path
from typing import Callable
from typing import Optional

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


VisitNodeFuncTypeDef = Callable[
    [list[tuple[str, set[str], Optional[str]]], str, set[str], Optional[str]], None
]


def visit_1(
    queue: list[tuple[str, set[str], str | None]],
    node: str,
    path: set[str],
    twice: str | None = None,
) -> None:
    if node.isupper() or node not in path:
        queue.append((node, {*path, node}, None))


def visit_2(
    queue: list[tuple[str, set[str], str | None]],
    node: str,
    path: set[str],
    twice: str | None = None,
) -> None:
    if node in path:
        if node.isupper():
            queue.append((node, {*path, node}, twice))
        else:  # only if we haven't visited any cave twice
            if twice is None:
                queue.append((node, {*path, node}, node))
    else:
        queue.append((node, {*path, node}, twice))


def compute(s: str, part: int) -> int:
    edges: list[tuple[str, str]] = []
    graph: dict[str, set[str]] = {}

    visit: VisitNodeFuncTypeDef = visit_1 if part == 1 else visit_2

    for edge in s.splitlines():
        start, end = edge.split("-")
        edges.append((start, end))
        if start not in graph:
            graph[start] = set()
        graph[start].add(end)

        if end not in graph:
            graph[end] = set()
        graph[end].add(start)

    queue: list[tuple[str, set[str], str | None]] = [("start", {"start"}, None)]

    result = 0

    while queue:
        node, path, twice = queue.pop()

        for child in graph[node]:
            if child == "end":
                result += 1
            elif child == "start":
                continue
            else:
                visit(queue, child, path, twice)

    return result


INPUT_S = """\
start-A
start-b
A-c
A-b
b-d
A-end
b-end
"""


INPUT_S_1 = """\
dc-end
HN-start
start-kj
dc-start
dc-HN
LN-dc
HN-end
kj-sa
kj-HN
kj-dc
"""

INPUT_S_2 = """\
fs-end
he-DX
fs-he
start-DX
pj-DX
end-zg
zg-sl
zg-pj
pj-he
RW-he
fs-DX
pj-RW
zg-RW
start-pj
he-WI
zg-he
pj-fs
start-RW
"""


@pytest.mark.parametrize(
    ("input_s", "part", "expected"),
    (
        (INPUT_S, 1, 10),
        (INPUT_S_1, 1, 19),
        (INPUT_S_2, 1, 226),
        (INPUT_S, 2, 36),
        (INPUT_S_1, 2, 103),
        (INPUT_S_2, 2, 3509),
    ),
)
def test(input_s: str, part: int, expected: int) -> None:
    assert compute(input_s, part) == expected


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
