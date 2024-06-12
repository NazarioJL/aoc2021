from __future__ import annotations

import argparse
import operator
import os.path
from collections import deque
from functools import reduce, cache
from itertools import combinations
from typing import cast, Callable, Iterable, TypeVar, Union

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


Point = tuple[int, int, int]

ROTATION_TRANSFORMATIONS: list[Callable[[Point], Point]] = [
    # rotates on the current x-axis in the direction of x (y cross z)
    lambda p: (p[0], p[1], p[2]),  # (x, y, z)
    lambda p: (p[0], -p[2], p[1]),  # (x, -z, y)
    lambda p: (p[0], -p[1], -p[2]),  # (x, -y, -z)
    lambda p: (p[0], p[2], -p[1]),  # (x, z, -y)
]

MAIN_AXIS_TRANSFORMATIONS: list[Callable[[Point], Point]] = [
    lambda p: (p[0], p[1], p[2]),  # (x, y, z) x is forward
    lambda p: (p[2], p[0], p[1]),  # (z, x, y) z is forward
    lambda p: (p[1], p[2], p[0]),  # (y, z, x) y is forward
]

MIRROR_TRANSFORMATIONS: list[Callable[[Point], Point]] = [
    lambda p: (p[0], p[1], p[2]),  # (x, y, z)
    lambda p: (-p[0], -p[1], p[2]),  # (-x, -y, z)
]


TRANSFORMATIONS_FUNCS_IDX: list[tuple[int, int, int]] = [
    (idx_a, idx_m, idx_r)
    for idx_a in range(len(MAIN_AXIS_TRANSFORMATIONS))
    for idx_m in range(len(MIRROR_TRANSFORMATIONS))
    for idx_r in range(len(ROTATION_TRANSFORMATIONS))
]

TOTAL_TRANSFORMATIONS = 24


@cache
def get_transformation_func(idx: int) -> Callable[[Point], Point]:
    idx_a, idx_m, idx_r = TRANSFORMATIONS_FUNCS_IDX[idx]
    func_axis = MAIN_AXIS_TRANSFORMATIONS[idx_a]
    func_mirror = MIRROR_TRANSFORMATIONS[idx_m]
    func_rot = ROTATION_TRANSFORMATIONS[idx_r]

    return lambda p: func_rot(func_mirror(func_axis(p)))


ORIENTATION_FUNCS = [get_transformation_func(i) for i in range(24)]


def transform_points(
    transformation_index: int, p: Point | list[Point]
) -> Point | list[Point]:
    func = get_transformation_func(transformation_index)

    if isinstance(p, tuple):
        return func(p)
    else:
        return [func(p_) for p_ in p]


def get_all_transformations(
    points: list[Point],
) -> Iterable[tuple[int, Iterable[Point]]]:
    return ((idx, transform_points(idx, points)) for idx in range(24))


def normalize(locations: list[Point]) -> list[Point]:
    offset = get_offset(locations)
    return [
        cast(Point, tuple((p[i] - offset[i]) for i in (0, 1, 2))) for p in locations
    ]


def get_offset(locations: list[Point]) -> Point:
    return cast(
        Point,
        reduce(lambda acc, e: tuple(min(acc[i], e[i]) for i in (0, 1, 2)), locations),
    )  # find min values so we can move the whole structure of points


def subtract_points(a: Point, b: Point) -> Point:
    return cast(Point, tuple(a[i] - b[i] for i in (0, 1, 2)))


def add_points(a: Point, b: Point) -> Point:
    return cast(Point, tuple(a[i] + b[i] for i in (0, 1, 2)))


def build_offsets(points: Iterable[Point]) -> dict[Point, set[Point]]:
    result = {}
    for p in points:
        result[p] = build_relative_offsets(p, points)

    return result


def build_relative_offsets(point: Point, points: Iterable[Point]) -> set[Point]:
    return {subtract_points(point, p) for p in points}


def read_beacons(s: str) -> list[list[Point]]:
    beacons: list[list[Point]] = []
    for idx, scanner_data in enumerate(s.split("\n\n")):
        current_list = []
        for line in scanner_data.splitlines()[1:]:
            x, y, z = line.split(",")
            current_list.append((int(x), int(y), int(z)))
        beacons.append(current_list)

    return beacons


def merge_beacons(
    reference_beacons: dict[Point, set[Point]], other_beacons: list[Point]
) -> tuple[Point, int] | bool:
    def _is_match(other_offsets: dict[Point, set[Point]]) -> tuple[Point, Point] | bool:
        # could not get this to work with 12 common points, 11 did work though
        for ref_point_, ref_offset in reference_beacons.items():
            for other_point_, other_offset in other_offsets.items():
                if len(ref_offset.intersection(other_offset)) >= 11:
                    return ref_point_, other_point_

        return False

    # produce all orientations for this beacon
    for idx, transformed_beacons in get_all_transformations(other_beacons):
        transformed_beacons_offsets = build_offsets(transformed_beacons)
        if match_result := _is_match(transformed_beacons_offsets):
            ref_point, other_point = match_result
            other_scanner_offset = subtract_points(ref_point, other_point)

            # now that we found the point we can return the offset and orientation index
            other_beacons_transformed = [
                add_points(transform_points(idx, b), other_scanner_offset)
                for b in other_beacons
            ]

            for other in other_beacons_transformed:
                if other not in reference_beacons:
                    # This point does not exist in the reference
                    reference_beacons[other] = build_relative_offsets(
                        other, reference_beacons.keys()
                    )

            return other_scanner_offset, idx

    return False


def distance(a: Point, b: Point) -> int:
    return sum(abs(a[i] - b[i]) for i in (0, 1, 2))


def compute(s: str) -> int:
    scanner_beacons: list[list[tuple[int, int, int]]] = read_beacons(s)

    # The first scanner is referenced at (0, 0, 0)
    scanner_locations: list[Point] = [(0, 0, 0)]
    transformations = []
    all_beacons_with_offsets: dict[Point, set[Point]] = build_offsets(
        scanner_beacons.pop(0)
    )

    queue = deque(
        reversed([(s_idx, b_lst) for s_idx, b_lst in enumerate(scanner_beacons)])
    )

    unmergable_count = 0

    while queue:
        scanner_idx, beacons = queue.pop()
        print(f"Attempting to merge scanner data: {scanner_idx + 1}")
        if merge_result := merge_beacons(all_beacons_with_offsets, beacons):
            new_offset, orientation = merge_result
            scanner_locations.append(new_offset)
            transformations.append(orientation)
            print(f"Successful merge, scanner {scanner_idx + 1} is at: {new_offset}")
            unmergable_count = 0
        else:
            print(f"No possible merge from scanner: {scanner_idx}")
            queue.appendleft((scanner_idx, beacons))
            unmergable_count += 1
            if unmergable_count == len(queue):
                raise ValueError(
                    f"Unable to merge any beacons, unmergable scanner "
                    f"counts = {len(queue)}"
                )

    max_distance = 0
    for a, b in combinations(scanner_locations, 2):
        max_distance = max(distance(a, b), max_distance)

    print(max_distance)  # This is part 2 no time to write tests

    return len(all_beacons_with_offsets)


INPUT_S = """\
--- scanner 0 ---
404,-588,-901
528,-643,409
-838,591,734
390,-675,-793
-537,-823,-458
-485,-357,347
-345,-311,381
-661,-816,-575
-876,649,763
-618,-824,-621
553,345,-567
474,580,667
-447,-329,318
-584,868,-557
544,-627,-890
564,392,-477
455,729,728
-892,524,684
-689,845,-530
423,-701,434
7,-33,-71
630,319,-379
443,580,662
-789,900,-551
459,-707,401

--- scanner 1 ---
686,422,578
605,423,415
515,917,-361
-336,658,858
95,138,22
-476,619,847
-340,-569,-846
567,-361,727
-460,603,-452
669,-402,600
729,430,532
-500,-761,534
-322,571,750
-466,-666,-811
-429,-592,574
-355,545,-477
703,-491,-529
-328,-685,520
413,935,-424
-391,539,-444
586,-435,557
-364,-763,-893
807,-499,-711
755,-354,-619
553,889,-390

--- scanner 2 ---
649,640,665
682,-795,504
-784,533,-524
-644,584,-595
-588,-843,648
-30,6,44
-674,560,763
500,723,-460
609,671,-379
-555,-800,653
-675,-892,-343
697,-426,-610
578,704,681
493,664,-388
-671,-858,530
-667,343,800
571,-461,-707
-138,-166,112
-889,563,-600
646,-828,498
640,759,510
-630,509,768
-681,-892,-333
673,-379,-804
-742,-814,-386
577,-820,562

--- scanner 3 ---
-589,542,597
605,-692,669
-500,565,-823
-660,373,557
-458,-679,-417
-488,449,543
-626,468,-788
338,-750,-386
528,-832,-391
562,-778,733
-938,-730,414
543,643,-506
-524,371,-870
407,773,750
-104,29,83
378,-903,-323
-778,-728,485
426,699,580
-438,-605,-362
-469,-447,-387
509,732,623
647,635,-688
-868,-804,481
614,-800,639
595,780,-596

--- scanner 4 ---
727,592,562
-293,-554,779
441,611,-461
-714,465,-776
-743,427,-804
-660,-479,-426
832,-632,460
927,-485,-438
408,393,-506
466,436,-512
110,16,151
-258,-428,682
-393,719,612
-211,-452,876
808,-476,-593
-575,615,604
-485,667,467
-680,325,-822
-627,-443,-432
872,-547,-609
833,512,582
807,604,487
839,-516,451
891,-625,532
-652,-548,-490
30,-46,-14
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 79),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def print_2d_board(points: list[tuple]) -> None:
    max_x = max(points, key=operator.itemgetter(0))[0]
    max_y = max(points, key=operator.itemgetter(1))[1]

    min_x = min(points, key=operator.itemgetter(0))[0]
    min_y = min(points, key=operator.itemgetter(1))[1]

    coords = set((p[0], p[1]) for p in points)

    min_y = min(min_y, 0)
    min_x = min(min_x, 0)

    max_y = max(max_y, 0)
    max_x = max(max_x, 0)

    for y in reversed(range(min_y, max_y + 1)):
        buffer = []
        for x in range(min_x, max_x + 1):
            if (x, y) in coords:
                buffer.append("B")
            elif (x, y) == (0, 0):
                buffer.append("S")
            else:
                buffer.append(".")
        print("".join(buffer))


def test_minimal_2d():
    # Original idea of offsetting to a common location
    beacons_1 = [(0, 2, 0), (4, 1, 0), (3, 3, 0)]

    beacons_2 = [(-1, -1, 0), (-5, 0, 0), (-2, 1, 0)]

    print("Beacon Set1")
    print_2d_board(beacons_1)

    print("Beacon Set2")
    print_2d_board(beacons_2)

    print("Beacon Set1 Normalized")
    print_2d_board(normalize(beacons_1))

    print("Beacon Set2 Normalized")
    print_2d_board(normalize(beacons_2))

    normalized_1 = set(normalize(beacons_1))
    normalized_2 = set(normalize(beacons_1))

    print(f"{len(normalized_1)=}")
    print(f"{len(normalized_2)=}")

    print(normalized_1.intersection(normalized_2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


TNode = TypeVar("TNode", bound=tuple)
Node = Union[tuple[int, TNode], tuple[TNode, int], tuple[TNode, TNode], tuple[int, int]]

v: Node = (1, (1, 2))

Number = Union[int, "Number"]
