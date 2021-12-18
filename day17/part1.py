from __future__ import annotations

import argparse
import os.path
import re
from math import sqrt
from typing import Callable
from typing import NamedTuple

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


class Point(NamedTuple):
    x: int
    y: int


class Velocity(NamedTuple):
    x: int
    y: int


class Probe(NamedTuple):
    position: Point
    velocity: Velocity


class Rectangle(NamedTuple):
    top_left: Point
    bottom_right: Point

    @property
    def top(self) -> int:
        return self.top_left.y

    @property
    def bottom(self) -> int:
        return self.bottom_right.y

    @property
    def left(self) -> int:
        return self.top_left.x

    @property
    def right(self) -> int:
        return self.bottom_right.x


def intersects(point: Point, rectangle: Rectangle) -> bool:
    return (
        rectangle.bottom <= point.y <= rectangle.top
        and rectangle.left <= point.x <= rectangle.right
    )


def sign(x):
    return (x > 0) - (x < 0)


StepFuncTypeDef = Callable[[Probe, int, int], Probe]


def step(probe: Probe, drag: int = 1, gravity: int = 1) -> Probe:
    new_pos_x = probe.position.x + probe.velocity.x
    new_pos_y = probe.position.y + probe.velocity.y

    new_vel_x = probe.velocity.x - sign(probe.velocity.x) * drag
    new_vel_y = probe.velocity.y - gravity

    return Probe(
        position=Point(x=new_pos_x, y=new_pos_y),
        velocity=Velocity(x=new_vel_x, y=new_vel_y),
    )


# target area: x=253..280, y=-73..-46
def find_highest(target: Rectangle, start: Point = Point(0, 0)) -> tuple[int, int]:
    max_y = 0
    count = 0
    for x in range(0, 280):
        for y in range(-73, 220):
            high_y = calc_path(
                initial_position=start,
                initial_velocity=Velocity(x=x, y=y),
                target=target,
                max_steps=200,
            )
            if high_y:
                count += 1
                max_y = max(max_y, high_y)
                # print(f"Collision at: {(x, y)}")

    print(f"Solution!!! {max_y=}, {count=}")
    return max_y, count


def calc_path(
    initial_position: Point,
    initial_velocity: Velocity,
    target: Rectangle,
    max_steps: int = 250,
) -> int | None:
    probe = Probe(position=initial_position, velocity=initial_velocity)
    max_y = 0
    for _ in range(max_steps):
        max_y = max(max_y, probe.position.y)
        if probe.velocity.x == 0 and probe.position.x < target.left:
            break
        if probe.velocity.y < 0 and probe.position.y < target.bottom:
            break
        if probe.position.y < target.bottom:
            break
        if intersects(probe.position, target):
            return max_y
        probe = step(probe)
    return None


def get_parameters(target: str) -> Rectangle:
    regex = (
        r"^target area: x=(?P<x1>-?\d+)..(?P<x2>-?\d+), y=(?P<y1>-?\d+)..(?P<y2>-?\d+)$"
    )
    match = re.match(regex, target)

    x1 = int(match.group("x1"))
    x2 = int(match.group("x2"))
    y1 = int(match.group("y1"))
    y2 = int(match.group("y2"))

    top_left = Point(min(x1, x2), max(y1, y2))
    bottom_right = Point(max(x1, x2), min(y1, y2))

    return Rectangle(top_left=top_left, bottom_right=bottom_right)


def get_x_intercepts(
    left: int, right: int, start: int = 0
) -> list[tuple[int, int, int]]:
    """
    Given a starting point and a bounding range, find all
    """

    min_velocity = get_min_velocity(left)
    max_velocity = right

    result = []
    for velocity in range(min_velocity, max_velocity + 1):
        # get the time t right before hitting the box
        total_distance = (velocity * velocity + velocity) // 2

        v = velocity
        pos = start
        t = 0

        while pos <= right:
            if pos >= left:
                result.append((t, velocity, pos))

            if velocity <= 0:
                break

            pos += v
            t += 1
            v -= 1

    return result


def get_min_velocity(x: int, start: int = 0) -> int:
    """
    Gets the minimum velocity to reach start when velocity is decreased by 1 on every
    step.

    0 --*--*-**----| x = 15
    0   1  2 34      initial velocity is 4 will reach a max of 10 and not get to x (15)

    0 --*--*-**----| x = 15
    0     1    2   3  4  5 67 initial velocity of 5 will reach and cross x on step 3

    We can solve the initial distance by solving the quadratic equation of the sum of
    consecutive integers x = (n * n + n) / 2. There are to n values that will solve this
    we only take the positive larger value.
    """

    return int((sqrt(8 * (x - start) + 1) - 1) / 2.0)  # return the positive solution


def compute(s: str) -> int:
    target = get_parameters(s)
    find_highest(target)

    return 0


INPUT_S = """\
target area: x=20..30, y=-10..-5
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1),),
)
def test(input_s: str, expected: int) -> None:
    assert compute(input_s) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
