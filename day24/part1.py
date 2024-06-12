from __future__ import annotations

import argparse
import os.path
from typing import Iterable

import pytest

from .lib import parse_program, ALU, Registers, ProgramCompressor, solve, get_info
from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


def compute(s: str, number: str = None) -> int:
    instructions = list(parse_program(s))
    print(f"Program has {len(instructions)} instructions")
    number = number or "89579246897766"
    if len(number) != 14:
        raise ValueError("Number MONAD has to be 14 digits")

    if "0" in number:
        raise ValueError("Number MONAD cannot have 0's")

    print(f"Input number is: {number}")

    alu = ALU(instructions, number=[int(n) for n in number])
    alu.run()
    print(alu.inspect().registers)
    # print(f"After run: {(val, val)=}")

    compressor = ProgramCompressor(parse_program(INPUT_VAR))
    compressor.run()
    for var, ex in compressor.inspect():
        # result = solve(ex, number=[int(n) for n in number])
        # print(f"{var=} -> {result=}")
        print(f"{var} = {get_info(ex)}")
        print(f"{var} = {ex}")

    prog: dict[int, list[tuple[str, str, str]]] = {}

    seg = -1
    for line in s.splitlines():
        parts = line.split()
        if parts[0] == "inp":
            seg += 1
            prog[seg] = []
        else:
            prog[seg].append(tuple(parts))

    a, b, c = [], [], []
    for idx, inst in enumerate(instructions):
        mod = idx % 18
        if mod == 4:
            a.append(inst.args[1])
            print(f"a={inst.args[1]}")
        if mod == 5:
            print(f"b={inst.args[1]}")
            b.append(inst.args[1])
        if mod == 15:
            print(f"c={inst.args[1]}")
            c.append(inst.args[1])

    print(a)
    print(b)
    print(c)
    print(", ".join(a))
    print(", ".join(b))
    print(", ".join(c))

    return 0


"""
[1, 1, 1, 26, 1, 1, 1, 26, 1, 26, 26, 26, 26, 26]
[14, 10, 13, -8, 11, 11, 14, -11, 14, -1, -8, -5, -16, -6]
[12, 9, 8, 3, 0, 11, 10, 13, 3, 10, 10, 14, 6, 5]
"""

INPUT_VAR = """\
add w n
mul x 0
add x z
mod x 26
div z a
add x b
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y c
mul y x
add z y
"""


INPUT_S = """\

"""

PROGRAM_1 = """\
inp x
mul x -1
"""

PROGRAM_2 = """\
inp z
inp x
mul z 3
eql z x
"""

PROGRAM_3 = """\
inp w
add z w
mod z 2
div w 2
add y w
mod y 2
div w 2
add x w
mod x 2
div w 2
mod w 2
"""

REAL_PROGRAM = """\
inp w
mul x 0
add x z
mod x 26
div z 1
add x 14
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 12
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 10
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 9
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 13
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 8
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -8
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 3
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 11
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 0
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 11
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 11
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 14
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 10
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -11
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 13
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 1
add x 14
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 3
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -1
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 10
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -8
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 10
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -5
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 14
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -16
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 6
mul y x
add z y
inp w
mul x 0
add x z
mod x 26
div z 26
add x -6
eql x w
eql x 0
mul y 0
add y 25
mul y x
add y 1
mul z y
mul y 0
add y w
add y 5
mul y x
add z y
"""

"""
z = (('n_0', add, 12), mul, ((0, eql, 'n_0'), eql, 0))
z = (('n_1', add, 9), mul, (((((('n_0', add, 12), mul, ((0, eql, 'n_0'), eql, 0)), mod, 26), add, 10), eql, 'n_1'), eql, 0))
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def test_simple_program():
    program = parse_program(PROGRAM_1)

    number = [1]
    alu = ALU(program=program, number=number)
    alu.run()
    state = alu.inspect()
    assert state.registers == (("w", 0), ("x", -1), ("y", 0), ("z", 0))


@pytest.mark.parametrize(
    ("program", "number", "expected"),
    (
        (PROGRAM_1, [1], (("w", 0), ("x", -1), ("y", 0), ("z", 0))),
        (PROGRAM_1, [-9], (("w", 0), ("x", 9), ("y", 0), ("z", 0))),
        (PROGRAM_2, [2, 6], (("w", 0), ("x", 6), ("y", 0), ("z", 1))),
        (PROGRAM_2, [2, 7], (("w", 0), ("x", 7), ("y", 0), ("z", 0))),
        (PROGRAM_3, [15], (("w", 1), ("x", 1), ("y", 1), ("z", 1))),
        (PROGRAM_3, [4], (("w", 0), ("x", 1), ("y", 0), ("z", 0))),
    ),
)
def test_alu_run(program: str, number: Iterable[int], expected: Registers):
    program = parse_program(program)
    alu = ALU(program, number)
    alu.run()

    state = alu.inspect()
    assert state.registers == expected


def test_expression():
    compressor = ProgramCompressor(program=parse_program(PROGRAM_3))
    compressor.run()
    state = compressor.inspect()

    assert state


def test_solve():
    x = solve(exp, number=[int(n) for n in "13579246899999"])
    assert x == 34


def test_real_program():
    instructions = list(parse_program(REAL_PROGRAM))
    compressor = ProgramCompressor(instructions[:72])
    compressor.run()

    assert compressor.inspect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    parser.add_argument("--n", type=str, default=None)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read(), args.n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
