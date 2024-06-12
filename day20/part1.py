from __future__ import annotations

import argparse
import os.path
from functools import reduce
from typing import Iterable

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")

Pixel = tuple[int, int]


def null_max(a: int | None, b: int) -> int:
    if not a:
        return b
    else:
        return max(a, b)


def null_min(a: int | None, b: int) -> int:
    if not a:
        return b
    else:
        return min(a, b)


class Image:
    def __init__(self):
        self._pixels: set[Pixel] = set()
        self._max_x = 0
        self._max_y = 0
        self._min_x = 0
        self._min_y = 0

    def __str__(self):
        return (
            f"Image(min_x={self._min_x}, max_x={self._max_x}, min_y={self._min_y}, "
            f"max_y={self._max_y}, raw_pixels=[..., count={len(self._pixels)}])"
        )

    def add_pixel(self, pixel: Pixel):
        self._min_x = null_min(self._min_x, pixel[0])
        self._max_x = null_max(self._max_x, pixel[0])
        self._min_y = null_min(self._min_x, pixel[1])
        self._max_y = null_max(self._max_x, pixel[1])
        self._pixels.add(pixel)

    @property
    def min_x(self):
        return self._min_x

    @property
    def max_x(self):
        return self._max_x

    @property
    def min_y(self):
        return self._min_y

    @property
    def max_y(self):
        return self._max_y

    @property
    def raw_pixels(self) -> set[Pixel]:
        return self._pixels



def print_image(img: Image) -> None:
    print()
    print(f"top_left=({img.min_x}, {img.min_y})")

    for y in range(img.min_y, img.max_y + 1):
        buffer = []
        for x in range(img.min_x, img.max_x + 1):
            if (x, y) in img.raw_pixels:
                buffer.append("#")
            else:
                buffer.append(".")
        print("".join(buffer))


def get_window(p: Pixel) -> Iterable[Pixel]:
    # fmt: off
    offsets = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
    )

    for off_x, off_y in offsets:
        yield p[0] + off_x, p[1] + off_y


def get_pixels_influenced(p: Pixel) -> Iterable:
    # this creates a 5 by 5 block
    # fmt: off
    offsets = (
        (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
        (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
    )

    for off_x, off_y in offsets:
        yield p[0] + off_x, p[1] + off_y


def enhance_image(image: Image, algo: str, seq: int) -> Image:
    new_image: Image = Image()
    pixels_influenced = set()

    for pixel in image.raw_pixels:
        for new_pixel in get_pixels_influenced(pixel):
            pixels_influenced.add(new_pixel)
    
    for pixel in pixels_influenced:
        window = get_window(pixel)
        number = [
            p_w in image.raw_pixels
            for p_w in window
        ]
        algo_index = reduce(lambda acc, b: acc * 2 + int(b), number)

        if algo[algo_index] == "#":
            new_image.add_pixel(pixel)
        else:
            pass

    return new_image


def compute(s: str) -> int:
    algo_data: str
    algo_data, image_data = s.split("\n\n")

    algorithm = algo_data.replace("\n", "")
    image: Image = Image()

    y = 0
    for line in image_data.splitlines():
        for x, c in enumerate(line):
            if c == "#":
                image.add_pixel((x, y))
        y += 1

    print_image(image)
    image = enhance_image(image, algorithm, 0)
    print_image(image)
    image = enhance_image(image, algorithm, 1)
    print_image(image)

    return len(image.raw_pixels)


INPUT_S = """\
..#.#..#####.#.#.#.###.##.....###.##.#..###.####..#####..#....#..#..##..###..######.###...####..#..#####..##..#.#####...##.#.#..#.##..#.#......#.###.######.###.####...#.##.##..#..#..#####.....#.#....###..#.##......#.....#..#..#..##..#...##.######.####.####.#.#...#.......#..#.#.#...####.##.#......#..#...##.#.##..#...##.#.##..###.#......#.#.......#.#.#.####.###.##...#.....####.#..#..#.##.#....##..#.####....##...##..#...#......#.#.......#.......##..####..#...#.#.#...##..#.#..###..#####........#..####......#..#

#..#.
#....
##..#
..#..
..###
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 35),),
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
