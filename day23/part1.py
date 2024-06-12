from __future__ import annotations

import argparse
import os.path
from _heapq import heappop, heappush
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

State = tuple[int, int, int, int, int, int, int, int, int]


"""
#############
#...........#
###B#C#B#D###
  #A#D#C#A#
  #########
#############
#...........#
###B#C#C#B###
  #D#D#A#A#
  #########
#############
#...........#
###B#C#C#B###
  #D#D#A#A#
  #########  
"""

initial_state = (
    8,  # A0
    14,  # A1
    7,  # B0
    11,  # B1
    9,  # C0
    12,  # C1
    10,  # D0
    13,  # D1
    0,  # Move state
)

part_1_initial_state = (
    12,
    14,
    7,
    13,
    9,
    11,
    8,
    10,
)

"""

A

"""

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

TYPE_TO_ROOM: dict[int, set[int]] = {
    0: {7, 8},
    1: {9, 0xA},
    2: {0xB, 0xC},
    3: {0xD, 0xE},
}


LEGAL_MOVES_COST = {
    # Maps a location to positions it can move to and related cost single step only
    # Back rooms
    8: ((7, 1),),
    10: ((9, 1),),
    12: ((11, 1),),
    14: ((13, 1),),
    # Front rooms
    7: ((1, 2), (2, 2), (8, 1)),
    9: ((2, 2), (3, 2), (0xA, 1)),
    11: ((3, 2), (4, 2), (0xC, 1)),
    13: ((4, 2), (5, 2), (0xE, 1)),
    # Corridor
    0: ((1, 1),),
    1: ((0, 1), (2, 2), (7, 2)),
    2: ((1, 2), (3, 2), (7, 2), (9, 2)),
    3: ((2, 2), (4, 2), (9, 2), (0xB, 2)),
    4: ((3, 2), (5, 2), (0xB, 2), (0xD, 2)),
    5: ((4, 2), (6, 1), (0xD, 2)),
    6: ((5, 1),),
}

ALL_ROOMS = {*range(7, 15)}  # All side rooms are [7, 15)

DISALLOWED_ROOMS = {idx: ALL_ROOMS - TYPE_TO_ROOM[idx] for idx in range(4)}


PAIRS = [1, 0, 3, 2, 5, 4, 7, 6]


def get_start_end_cost() -> dict[tuple[int, int], int]:
    cost_table: dict[tuple[int, int], int] = {}

    for start in range(15):
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


def move_amp(state: State, amp: int, new_loc: int, move_state: int) -> State:
    """
    Moves amp to new_loc and creates new state
    """
    if 0 > amp >= 8:
        raise ValueError(f"{amp=} is out of range")

    return cast(
        State,
        tuple(
            [
                *(new_loc if idx == amp else loc for idx, loc in enumerate(state[:8])),
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
    occupancy = {loc: amp for amp, loc in enumerate(state[:8])}
    move_state = state[-1]

    for amp, curr_loc in enumerate(state[:8]):
        bit_pos = 1 << amp  # one of 2*i | i in [0..7]
        has_moved = bool(bit_pos & move_state)
        amp_type = amp // 2  # one of (0, 1, 2, 4)
        cost_mult = 10 ** amp_type  # one of (1, 10, 100, 1000)

        # Get the final destination rooms for these amps
        back_room_dest = amp_type * 2 + 8  # one of (8, 10, 12, 14)
        front_room_dest = back_room_dest - 1  # one of (7, 9, 11, 13)

        # Analyze if is in final destination
        if curr_loc in (back_room_dest, front_room_dest):
            # We can be in any of these cases when already in our destination room
            #
            #                   -------------------------
            #  current_amp = O  =========Hallway=========
            #  same_type   = Y  +---+---+---+---+---+---+
            #  other_type  = X  | O | O | O | Y | X |   |  -> front_room
            #                   +---+---+---+---+---+---+
            #                   | Y | X |   | O | O | O |  -> back_room
            #                   +---+---+---+---+---+---+
            #   case:             1   2   3   4   5   6
            #
            #   1: Continue, final state, consider next amp
            #   2: Consider moves, X has to get out
            #   3: Move to back_room_dest! consider next amp
            #   4: Continue, final state, consider next amp
            #   5: Do nothing, obstructed anyways, final state, next amp
            #   6: Do nothing, final state, next amp
            if curr_loc == back_room_dest:  # cases 4-6
                continue
            else:
                if back_room_dest not in occupancy:  # case 3
                    # Please move in!
                    yield move_amp(
                        state, amp, back_room_dest, move_state & bit_pos
                    ), cost_mult, (amp, back_room_dest)
                    continue
                else:
                    if PAIRS[amp] == occupancy[back_room_dest]:  # case 1
                        continue  # skip to next amp, ideal position here as well
                    else:  # ugh, someone is blocked in my burrow case 2
                        pass  # consider moves for this amp
        # get possible moves
        moves = get_possible_moves(curr_loc, occupancy)

        # optimistically try to go to final spot
        if back_room_dest in moves:
            yield move_amp(
                state, amp, back_room_dest, move_state & bit_pos
            ), cost_mult * moves[back_room_dest], (amp, back_room_dest)
            continue
        if front_room_dest in moves and occupancy[back_room_dest] == PAIRS[amp]:
            yield move_amp(
                state, amp, front_room_dest, move_state & bit_pos
            ), cost_mult * moves[front_room_dest], (amp, front_room_dest)
            continue

        if has_moved:
            # We could not move into our side_room with our pair at this point, this
            # means it is occupied
            continue
        for move, cost in moves.items():
            if move in DISALLOWED_ROOMS[amp_type]:
                continue
            # At this point it is not possible that we can consider our destination room
            yield move_amp(state, amp, move, move_state & bit_pos), cost_mult * cost, (
                amp,
                move,
            )


def is_final_state(state: State) -> bool:
    for amp, loc in enumerate(state[:8]):
        amp_type = amp // 2
        if loc not in TYPE_TO_ROOM[amp_type]:
            return False

    return True


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
