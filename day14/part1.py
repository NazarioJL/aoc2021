from __future__ import annotations

import argparse
import os.path

import pytest
from more_itertools import pairwise

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def add_frequencies(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for k, v in a.items():
        result[k] = v + b.get(k, 0)

    for k, v in b.items():
        if k not in result:
            result[k] = v

    return result


def pair_to_frequency(a: str, b: str) -> dict[str, int]:
    result = {a: 1}
    if a == b:
        result[a] += 1
    else:
        result[b] = 1
    return result


def get_pair_grow_frequency(
    seed_pair: tuple[str, str],
    iterations: int,
    rules: dict[tuple[str, str], str],
    prev_memo: dict[tuple[str, str, int, bool], dict[str, int]] | None,
    left: bool = True,
) -> dict[str, int]:
    s_l, s_r = seed_pair
    memo: dict[tuple[str, str, int, bool], dict[str, int]] = prev_memo or {}

    def grow_single_rec(
        l: str, r: str, n: int, s: bool  # noqa: E741
    ) -> dict[str, int]:
        if n == iterations:
            if s:  # True is for the left result
                return pair_to_frequency(l, r)
            else:
                return {r: 1}

        if (l, r, n, s) in memo:
            return memo[(l, r, n, s)]

        ins = rules[(l, r)]
        left_l, left_r = (l, ins)
        right_l, right_r = (ins, r)

        left_result = grow_single_rec(left_l, left_r, n + 1, s)
        right_result = grow_single_rec(right_l, right_r, n + 1, False)

        tmp_result = add_frequencies(left_result, right_result)
        memo[(l, r, n, s)] = tmp_result

        return add_frequencies(left_result, right_result)

    result = grow_single_rec(s_l, s_r, 0, left)

    return result


def grow(seed: str, rules: dict[tuple[str, str], str]) -> str:
    new_polymer = []
    last = None
    for a, b in pairwise(seed):
        ins = rules[(a, b)]
        new_polymer.append(a)
        new_polymer.append(ins)
        last = b

    new_polymer.append(last)
    return "".join(new_polymer)


def compute(s: str, iterations: int) -> int:
    lines = s.splitlines()
    template = lines[0]

    rules: dict[tuple[str, str], str] = {}

    for line in lines[2:]:
        pair, ins = line.split(" -> ")
        rules[(pair[0], pair[1])] = ins

    reusable_memo: dict[tuple[str, str, int, bool], dict[str, int]] = {}

    is_leftmost = True
    result: dict[str, int] = {}

    for left, right in pairwise(template):
        pair_frequency = get_pair_grow_frequency(
            seed_pair=(left, right),
            iterations=iterations,
            rules=rules,
            prev_memo=reusable_memo,
            left=is_leftmost,
        )
        is_leftmost = False
        result = add_frequencies(result, pair_frequency)

    print(result)
    return max(result.values()) - min(result.values())


INPUT_S = """\
NNCB

CH -> B
HH -> N
CB -> H
NH -> C
HB -> C
HC -> B
HN -> C
NN -> C
BH -> H
NC -> B
NB -> B
BN -> B
BB -> N
BC -> B
CC -> N
CN -> C
"""


@pytest.mark.parametrize(
    ("input_s", "iterations", "expected"),
    ((INPUT_S, 10, 1588), (INPUT_S, 40, 2188189693529)),
)
def test(input_s: str, iterations: int, expected: int) -> None:
    assert compute(input_s, iterations) == expected


@pytest.mark.parametrize(
    ("iterations", "expected"),
    ((1, "ABA"), (2, "AABBA"), (3, "ABAABBBBA")),
)
def test_grow(iterations: int, expected: str) -> None:
    test_rules = {
        ("A", "A"): "B",
        ("A", "B"): "A",
        ("B", "A"): "B",
        ("B", "B"): "B",
    }

    seed = "AA"
    for i in range(iterations):
        seed = grow(seed, test_rules)

    assert seed == expected


@pytest.mark.parametrize(
    ("iterations", "expected"),
    (
        (0, {"A": 2}),
        (1, {"A": 2, "B": 1}),
        (2, {"A": 3, "B": 2}),
        (3, {"A": 4, "B": 5}),
    ),
)
def test_grow_single(iterations: int, expected: dict[str, int]) -> None:
    test_rules = {
        ("A", "A"): "B",
        ("A", "B"): "A",
        ("B", "A"): "B",
        ("B", "B"): "B",
    }
    result = get_pair_grow_frequency(("A", "A"), iterations, test_rules, {})
    assert result == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
