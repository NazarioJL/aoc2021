from __future__ import annotations

import operator
from collections import Counter
from copy import deepcopy
from enum import Enum, auto
from itertools import count, repeat
from typing import NamedTuple, Union, Iterable, Callable, cast, Iterator


def curry(g, f):
    def h(*args, **kwargs):
        return g(f(*args, **kwargs))

    return h


Const = int
Var = str

Value = Union[Const, Var]

Unary = tuple[Value]
Binary = tuple[Value, Value]
Args = Union[Unary, Binary]

Registers = dict[str, int]  # Ok for now


class OperationType(Enum):
    MUL = "mul"
    INP = "inp"
    ADD = "add"
    MOD = "mod"
    DIV = "div"
    EQL = "eql"

    def __repr__(self):
        return self.value


OPERATOR_MAP = {
    OperationType.MUL: operator.mul,
    OperationType.ADD: operator.add,
    OperationType.DIV: operator.floordiv,
    OperationType.MOD: operator.mod,
    OperationType.EQL: curry(int, operator.eq),  # coerce to int
    OperationType.INP: lambda a, b: b,  # b replaces a
}


def resolve(arg: Const | Var, registers: Registers) -> Const:
    return registers[arg] if isinstance(arg, str) else arg


OperationFuncTypeDef = Callable[[Args, Registers], Const]


UNARY_OPERATIONS = {OperationType.INP}

BINARY_OPERATIONS = {
    OperationType.EQL,
    OperationType.DIV,
    OperationType.MOD,
    OperationType.ADD,
    OperationType.MUL,
}


class Instruction(NamedTuple):
    op_type: OperationType
    args: Unary | Binary

    def __str__(self):
        return f"{str(self.op_type)} {self.args}"


def parse_program(program: str) -> list[Instruction]:
    for line in program.splitlines():
        parts = line.split()
        operation_type = OperationType(parts[0])

        if len(parts) == 2:  # unary
            yield Instruction(op_type=operation_type, args=(parts[1],))
        else:
            cast_fun = (
                str if parts[2] in ("w", "x", "y", "z", "a", "b", "c", "n") else int
            )
            yield Instruction(
                op_type=operation_type, args=(parts[1], cast_fun(parts[2]))
            )


class ALUError(Exception):
    pass


class ALUExecutionState(Enum):
    Idle = auto()
    Loaded = auto()
    Running = auto()
    Finished = auto()


class ALUStateInfo(NamedTuple):
    instruction_pointer: int
    registers: Registers
    last_number: int | None
    last_number_index: int | None


class ALU:
    def __init__(
        self,
        program: list[Instruction],
        number: Iterable[int],
        registers: Registers | None = None,
    ):
        self._registers: Registers = registers or {"w": 0, "x": 0, "y": 0, "z": 0}
        self._program: list[Instruction] = program
        self._pointer = 0
        self._execution_state = ALUExecutionState.Loaded
        self._number: Iterator[int] = iter(number)
        self._number_index = -1
        self._last_number: int | None = None

    def step(self):
        if self._pointer >= len(self._program):
            raise ALUError("No more instructions to execute")

    def _execute(self, instruction: Instruction) -> None:
        self._execution_state = ALUExecutionState.Running
        args = instruction.args
        register = args[0]  # First arg is ALWAYS the destination register
        lh = self._registers[register]

        if instruction.op_type == OperationType.INP:
            # Special case, 2nd arg comes from the number
            rh = next(self._number)
            self._last_number = rh
            self._number_index += 1
        else:
            rh = resolve(args[1], self._registers)

        self._registers[register] = OPERATOR_MAP[instruction.op_type](lh, rh)

    def inspect(self) -> ALUStateInfo:
        return ALUStateInfo(
            instruction_pointer=self._pointer,
            registers=cast(
                Registers, tuple((r, v) for r, v in self._registers.items())
            ),
            last_number=self._last_number,
            last_number_index=self._number_index,
        )

    def run(self):
        for line, instruction in enumerate(self._program, start=self._pointer):
            self._execute(instruction)
            self._pointer = line


Terminal = Union[Const, Var]
NodeType = Union[tuple, Terminal]
Expression = Union[tuple[NodeType, OperationType, NodeType], Terminal]


def copy_expression(expression: Expression) -> Expression:
    def copy_rec(e: Expression):
        if isinstance(e, tuple):
            l, o, r = e
            return copy_rec(l), o, copy_rec(r)
        else:
            return e

    return copy_rec(expression)


class ProgramCompressor:
    def __init__(self, program: list[Instruction]):
        self._register_versions = {"w": 0, "x": 0, "y": 0, "z": 0}
        self._register_expressions: dict[str, Expression] = {
            "w": 0,
            "x": 0,
            "y": 0,
            "z": 0,
        }
        self._program: list[Instruction] = program
        self._pointer = 0
        self._execution_state = ALUExecutionState.Loaded
        self._number: Iterator[int] = count()
        self._number_index = -1
        self._last_number: int | None = None
        self._optimizations = 0
        self._evaluations = 0

    def _execute(self, instruction: Instruction) -> None:
        op = instruction.op_type
        args = instruction.args
        register = args[0]  # First arg is ALWAYS the destination register
        lh = self._register_expressions[register]  # this current register value

        if op == OperationType.INP:
            n = next(self._number)
            var_name = f"n_{n}"
            self._number_index += 1
            self._last_number = n
            self._register_expressions[register] = var_name
        else:
            assert len(args) == 2

            if (
                isinstance(args[1], str) and args[1] in "wxyz"
            ):  # this refers to another expression
                rh = self._register_expressions[args[1]]
            else:
                rh = args[1]  # must be a literal number value

            # are both numbers we can resolve now and prune the tree
            # if isinstance(rh, int) and isinstance(lh, int):
            #     val = OPERATOR_MAP[op](lh, rh)
            #     self._register_expressions[register] = val
            #     self._optimizations += 1
            #     self._evaluations += 1
            #     return
            #
            # # Easy optimizations:
            # if op == OperationType.MUL:
            #     # Mult by 0 is 0
            #     if isinstance(rh, int) and rh == 0:
            #         self._register_expressions[register] = 0
            #         self._optimizations += 1
            #         return
            #     if isinstance(lh, int) and lh == 0:
            #         self._register_expressions[register] = 0
            #         self._optimizations += 1
            #         return
            #
            #     # LH * 1 == LH
            #     if isinstance(rh, int) and rh == 1:
            #         # lh stays the same value noop
            #         self._optimizations += 1
            #
            #         return
            #
            #     # 1 * RH == RH
            #     if isinstance(lh, int) and lh == 1:
            #         self._optimizations += 1
            #         self._register_expressions[register] = rh
            #         return
            #
            # if op == OperationType.ADD:
            #     # Add 0 is 0 (identity for sum)
            #     if isinstance(rh, int) and rh == 0:
            #         # do nothing lh statys the same A + 0 == A
            #         return
            #
            #     if isinstance(lh, int) and lh == 0:  # 0 + A == A
            #         self._register_expressions[register] = rh
            #         return
            #
            # if op == OperationType.DIV:
            #     if isinstance(rh, int) and rh == 1:  # A / 1 == A leave rh the same
            #         return

            # just set the new tree at register the instruction is operating on
            self._register_expressions[register] = (lh, instruction.op_type, rh)

    def run(self):
        for line, instruction in enumerate(self._program, start=self._pointer):
            self._execute(instruction)
            self._pointer = line

    @property
    def optimizations_count(self):
        return self._optimizations

    def inspect(self) -> tuple[tuple[str, Expression]]:
        return tuple((r, e) for r, e in self._register_expressions.items())


add = OperationType.ADD
mul = OperationType.MUL
mod = OperationType.MOD
div = OperationType.DIV
eql = OperationType.EQL


def solve(expression: Expression, number: Iterable[int]) -> int:
    # Build num
    var_lookup = {f"n_{idx}": v for idx, v in enumerate(number)}

    def solve_rec(expr: Expression) -> int:
        if isinstance(expr, str):
            return var_lookup[expr]
        if isinstance(expr, int):
            return expr
        else:
            if not isinstance(expr, tuple):
                raise Value(f"Expected to be tuple")
            if len(expr) != 3:
                raise Value(f"Expected to be tuple to have 3 values")
            l, o, r = expr
            return OPERATOR_MAP[o](solve_rec(l), solve_rec(r))

    return solve_rec(expression)


def find_number(expression: Expression, num_length: int) -> list[int]:
    def find_number_rec(
        ex: Expression, req: int | None, lookup: list[int]
    ) -> Iterable[list[int]]:
        n = [repeat(0, num_length)]
        if req:  # this expression must eval to req
            if isinstance(ex, int):
                if ex == req:
                    yield lookup
                else:
                    return
            elif isinstance(ex, str):
                # Do we have a value for this yet
                idx = int(ex[2:])
                if lookup[idx] == 0:
                    # We don't have a value for this n yet, try all from biggest to
                    # smallest
                    for i in reversed(range(1, 10)):
                        new_lookup = [*lookup]
                        new_lookup[idx] = i
                        yield new_lookup

            else:  # this is an expression can we make it 0 ever
                # l, o, r = ex
                # if o == OperationType.DIV:
                #
                # elif o == OperationType.MUL:
                # elif o == OperationType.ADD:
                # elif o == OperationType.EQL:
                # elif o == elif o == OperationType.MOD
                pass

    return []


def get_info(expression: Expression) -> tuple[int, int]:
    max_depth = 0
    node_count = 0

    def get_depth_rec(e: Expression, d: int):
        nonlocal max_depth
        nonlocal node_count
        max_depth = max(max_depth, d)
        node_count += 1
        if isinstance(e, tuple):
            l, _, r = e
            get_depth_rec(l, d + 1)
            get_depth_rec(r, d + 1)

    get_depth_rec(expression, 0)

    return max_depth, node_count
