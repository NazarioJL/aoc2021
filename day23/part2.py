from __future__ import annotations

import argparse
import os.path
from _heapq import heappop, heappush
from functools import reduce
from typing import Iterable, cast

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")

DEBUG = True


"""
State Model:

The state of the game is described by a 9-tuple of ints. The position of each tuple 
maps to each of the Amphipods, the values are the location of each Amphipod starting 
from A - D like so: A_0 -> 0, A_1 -> 2, ... D2 -> 7

The last tuple value is the state that says which Amphipod has already moved. This can
be encoded by a single bit per amphipod.

A0 A1 B0 B1 C0 C1 D0 D1 -> 11111110 (254) all Amphipods have moved at least once except
for D1

"""

# FourLocations must be sorted to minimize produced states

FourLocations = tuple[int, int, int, int]
StateV1 = tuple[FourLocations, FourLocations, FourLocations, FourLocations, int]


"""
State is represented by 4 + 1 integers. The first four correspond to the bits set on 
location by amp type (0, 1, 2, 3) -> (A, B, C, D)
"""
State = tuple[int, int, int, int, int]


LOCATION_TO_BIT = {idx: 1 << idx for idx in range(23)}


"""
#############
#...........#
###B#C#C#B###
  #D#D#A#A#
  #D#D#A#A#
  #B#C#C#B#
  #########
"""

initial_state_v1 = ((0, 9, 12, 14), (3, 5, 10, 11), (6, 7, 8, 13), (1, 2, 4, 15), 0)

part_1_initial_state_v1 = (
    (9, 10, 13, 14),
    (0, 4, 12, 15),
    (4, 7, 8, 11),
    (1, 2, 5, 6),
    0,
)


def v1_to_v2(state_v1: StateV1) -> State:
    result = []

    for amps in state_v1[:4]:
        s = 0
        for amp in amps:
            s |= LOCATION_TO_BIT[amp]
        result.append(s)

    result.append(state_v1[4])
    return cast(State, tuple(result))


def state_to_locations(state: int) -> Iterable[int]:
    for i in range(23):
        if state & LOCATION_TO_BIT[i]:
            yield i


def state_to_occupancy(state: State) -> dict[int, int]:
    result = {}
    for amp_type, st in enumerate(state[:4]):
        for loc in range(23):
            if st & LOCATION_TO_BIT[loc]:
                result[loc] = amp_type

    return result


"""
'        101001000000001'
321098765432109876543210
"""
"""

A

"""

"""
Hallway (.)
----------- 
...........
01x2x3x4x56
  7 A D G
  8 B E H
  9 C F I
-----------
  0 1 2 3  -> room/amphipod type
"""

TYPE_TO_ROOM: dict[int, set[int]] = {
    0: {0, 1, 2, 3},
    1: {4, 5, 6, 7},
    2: {8, 9, 10, 11},
    3: {12, 13, 14, 15},
}


LEGAL_MOVES_COST = {
    # Maps a location to positions it can move to and related cost single step only
    # Back rooms
    0: ((1, 1),),
    4: ((5, 1),),
    8: ((9, 1),),
    12: ((13, 1),),
    # Middle rooms
    1: (
        (0, 1),
        (2, 1),
    ),
    5: (
        (0, 4),
        (6, 1),
    ),
    9: (
        (0, 8),
        (10, 1),
    ),
    13: (
        (0, 14),
        (15, 1),
    ),
    2: (
        (1, 1),
        (3, 1),
    ),
    6: (
        (5, 1),
        (7, 1),
    ),
    10: (
        (9, 1),
        (11, 1),
    ),
    14: (
        (13, 1),
        (15, 1),
    ),
    # Front rooms
    3: ((2, 1), (17, 2), (18, 2)),
    7: ((6, 1), (18, 2), (19, 2)),
    11: (
        (10, 1),
        (19, 2),
        (20, 2),
    ),
    15: ((14, 1), (20, 2), (21, 2)),
    # Hallway
    16: ((17, 1),),
    17: (
        (16, 1),
        (3, 2),
        (18, 2),
    ),
    18: (
        (17, 1),
        (3, 2),
        (7, 2),
        (19, 2),
    ),
    19: (
        (18, 1),
        (7, 2),
        (11, 2),
        (20, 2),
    ),
    20: (
        (19, 1),
        (11, 2),
        (15, 2),
        (21, 2),
    ),
    21: (
        (20, 1),
        (15, 2),
        (22, 1),
    ),
    22: ((21, 1),),
}


ALL_ROOMS = {*range(16)}  # All side rooms are [7, 19)

DISALLOWED_ROOMS = {idx: ALL_ROOMS - TYPE_TO_ROOM[idx] for idx in range(4)}


PAIRS = [1, 0, 3, 2, 5, 4, 7, 6]


def get_start_end_cost() -> dict[tuple[int, int], int]:
    cost_table: dict[tuple[int, int], int] = {}

    for start in range(23):
        cost_table[(start, start)] = 0
        queue = [(start, 0)]
        while queue:
            node, cost = queue.pop()
            cost_table[(start, node)] = cost

            for child, child_cost in LEGAL_MOVES_COST[node]:
                new_cost = child_cost + cost
                # child_cost is the cost from node to child
                # cost is the cost from start to node

                prev_cost = cost_table.get((start, child))
                if prev_cost is None or new_cost < prev_cost:
                    queue.append((child, new_cost))

    return cost_table


# Lookup from any 2 locations to get cost
COST_TABLE = get_start_end_cost()


"""
Amphipod type:

Amber -> 0
Bronze -> 1
Copper -> 2
Desert -> 3

Amphipods indexes:

A1 -> 0 (Amber)
A2 -> 1
B1 -> 2 (Bronze)
B2 -> 3
C1 -> 4 (Copper)
C2 -> 5
D1 -> 6 (Desert)
D2 -> 7
"""


def is_room_occupied_by_others(state: State, room_type: int) -> bool:
    """Check if side room has amps whose final destination is not that room"""
    rooms = TYPE_TO_ROOM[room_type]  # 2 rooms per room_type

    for amp, loc in enumerate(state[:8]):
        if loc in rooms and (amp // 2) != room_type:
            return True

    return False


def get_all_moves(loc: int, occupied: dict[int, int]) -> Iterable[tuple[int, int]]:
    """
    Given the map this will generate all moves from the given location with associated
    costs, it will not return occupied locations. This uses LEGAL_MOVES_COST to
    calculate the paths.
    """

    visited: set[int] = {loc}
    queue = [(loc, 0)]

    while queue:
        l, c = queue.pop()
        if l not in occupied and l != loc:
            yield l, c
        for n_l, n_c in LEGAL_MOVES_COST[l]:
            if n_l not in visited:
                visited.add(n_l)
                queue.append((n_l, c + n_c))


def validate_state(state: State) -> bool:
    return len(set(state[:8])) == 8


def move_amp(
    state: State, amp_type: int, prev_loc: int, new_loc: int, move_state: int
) -> State:
    """
    Moves amp to new_loc and creates new state
    """

    if new_loc > 22:
        raise ValueError(f"{new_loc=} out of range...")

    # get specific section
    state_to_mutate = state[amp_type]
    state_to_mutate ^= LOCATION_TO_BIT[prev_loc]  # clear bit
    state_to_mutate |= LOCATION_TO_BIT[new_loc]

    return cast(
        State,
        tuple(
            [
                *[
                    state_to_mutate if idx == amp_type else amp_type_state
                    for idx, amp_type_state in enumerate(state[:4])
                ],
                move_state,
            ]
        ),
    )


def is_obstructed(amp: int, amp_loc: int, occupancy: dict[int, int]) -> bool:
    # amp can only be obstructed if at any other burrow
    raise NotImplementedError


def get_possible_moves(amp_loc: int, occupancy: dict[int, int]) -> dict[int, int]:
    results: dict[int, int] = {}

    def build_moves(visited: set[int], loc: int, cost: int) -> None:
        for l, c in LEGAL_MOVES_COST[loc]:
            if l in visited or l in occupancy or l == amp_loc:
                continue
            else:
                new_cost = cost + c
                prev_cost = results.get(l)
                if prev_cost is None or new_cost < prev_cost:
                    results[l] = new_cost
                    build_moves({*visited, l}, l, new_cost)

    build_moves(set(), amp_loc, 0)

    return results


def get_possible_states(state: State) -> Iterable[tuple[State, int, tuple[int, int]]]:
    """Returns all valid moves with associated costs"""
    # map of room to amp e.g. A (10) -> 0 (A1)
    occupancy = state_to_occupancy(state)

    # We care about exp

    move_state = state[-1]


def is_final_state(state: State) -> bool:
    return reduce(lambda acc, e: acc | e, state[:4]) == 0b1111_1111_1111_1111


def find_min_cost(state: State) -> tuple[int, State]:
    heap: list[tuple[int, state]] = [(0, state)]
    # map of state to cost for that state
    visited: dict[State, int] = {state: 0}

    while heap:
        cost, curr_state = heappop(heap)
        if is_final_state(curr_state):
            return cost, curr_state

        for next_state, next_cost, (amp, move_to) in get_possible_states(curr_state):
            new_cost = cost + next_cost
            # Have we seen this state before?
            prev_cost = visited.get(next_state)
            if prev_cost is None or new_cost < prev_cost:
                visited[next_state] = new_cost
                heappush(heap, (new_cost, next_state))

    raise Exception("Ran out states to evaluate")


def compute(s: str) -> int:
    # cheating on this one for reading input

    cost, final = find_min_cost(part_1_initial_state)

    return cost


INPUT_S = """\

"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def test_one_more_step() -> None:
    last_to_one = (8, 5, 10, 9, 11, 12, 13, 14, 0)
    cost, _ = find_min_cost(last_to_one)

    assert cost == 8


def test_two_more_steps() -> None:
    last_to_one = (8, 5, 10, 9, 11, 12, 3, 4, 0)
    cost, _ = find_min_cost(last_to_one)

    assert cost == 7008


def test_find_min_cost():
    cost, _ = find_min_cost(initial_state)
    assert cost == 12521


@pytest.mark.parametrize(
    ("state", "expected"),
    (
        (
            (
                7,  # A0
                8,  # A1
                9,  # B0
                10,  # B1
                11,  # C0
                12,  # C1
                13,  # D0
                14,  # D1
                0,  # Move state
            ),
            True,
        ),
        (
            (
                8,  # A0
                7,  # A1
                9,  # B0
                10,  # B1
                12,  # C0
                11,  # C1
                14,  # D0
                13,  # D1
                0,  # Move state
            ),
            True,
        ),
        (
            (
                7,  # A0
                9,  # A1
                8,  # B0
                10,  # B1
                11,  # C0
                12,  # C1
                13,  # D0
                14,  # D1
                0,  # Move state
            ),
            False,
        ),
        (
            (
                7,  # A0
                8,  # A1
                12,  # B0
                10,  # B1
                11,  # C0
                9,  # C1
                13,  # D0
                14,  # D1
                0,  # Move state
            ),
            False,
        ),
    ),
)
def test_in_final_state(state: State, expected: bool) -> None:
    assert is_final_state(state) is expected


def test_v1_v2():
    result = v1_to_v2(initial_state_v1)
    assert result


def test_find_all_moves():
    actual = set(get_all_moves(10, {}))
    actual_locs = set(a for a, _ in actual)
    all_locs = set(range(0, 15))
    all_locs.discard(10)

    assert actual_locs == all_locs

    assert actual == {
        (9, 1),
        (2, 3),
        (3, 3),
        (1, 5),
        (0, 6),
        (7, 5),
        (8, 6),
        (11, 5),
        (12, 6),
        (4, 5),
        (5, 7),
        (6, 8),
        (13, 7),
        (14, 8),
    }


def test_find_all_moves_with_occupied():
    actual = set(get_all_moves(10, {1: 2, 3: 4}))

    assert actual == {
        (9, 1),
        (2, 3),
        (0, 6),
        (7, 5),
        (8, 6),
        (11, 5),
        (12, 6),
        (4, 5),
        (5, 7),
        (6, 8),
        (13, 7),
        (14, 8),
    }


# [(10, 0), (3, 3), (11, 5), (4, 5), (13, 7), (5, 7), (6, 8), (2, 3), (7, 5), (1, 5),
# (0, 6)]


"""
   Hallway (.)
----------- 
...........
01x2x3x4x56
  7 9 B D
  8 A C E
-----------
  0 1 2 3  -> room/amphipod type

"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
