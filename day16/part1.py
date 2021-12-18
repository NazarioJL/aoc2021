from __future__ import annotations

import argparse
import os.path
from dataclasses import dataclass
from enum import auto
from enum import Enum
from functools import reduce
from operator import lt
from typing import Callable
from typing import Iterable
from typing import NamedTuple
from typing import TypeVar
from typing import Union

import pytest

from support import timing

INPUT_TXT = os.path.join(os.path.dirname(__file__), "input.txt")


HEX_TO_BIN: dict[str, str] = {
    "0": "0000",
    "1": "0001",
    "2": "0010",
    "3": "0011",
    "4": "0100",
    "5": "0101",
    "6": "0110",
    "7": "0111",
    "8": "1000",
    "9": "1001",
    "A": "1010",
    "B": "1011",
    "C": "1100",
    "D": "1101",
    "E": "1110",
    "F": "1111",
}


class ParseError(Exception):
    pass


class PacketType(Enum):
    Literal = auto()
    Operator = auto()
    Pad = auto()
    Unknown = auto()


@dataclass(frozen=True)
class PacketHeader:
    packet_version: str
    packet_type_id: str
    packet_type: PacketType

    @property
    def is_terminal(self) -> bool:
        return self.packet_type == PacketType.Literal


@dataclass
class PacketLiteral:
    header: PacketHeader
    value: str
    range: tuple[int, int]
    data: str | None = None
    packet_type: PacketType = PacketType.Literal


@dataclass
class PacketPad:
    range: tuple[int, int]
    data: str | None = None
    packet_type: PacketType = PacketType.Pad


class PacketLengthTypeId(Enum):
    TotalLengthOfSubPackets = "0"
    NumberOfSubPacketsContained = "1"


@dataclass
class Packet:
    header: PacketHeader
    length_type_id: PacketLengthTypeId
    range: tuple[int, int]
    children: list[Packet | PacketLiteral | PacketPad]
    data: str | None = None
    packet_type: PacketType = PacketType.Operator


def get_header(data: str, start: int) -> PacketHeader:
    end_version = start + 3
    end_type = start + 6
    version = data[start:end_version]
    type_id = data[end_version:end_type]
    p_type = get_packet_type(type_id)

    return PacketHeader(
        packet_version=version,
        packet_type=p_type,
        packet_type_id=type_id,
    )


def packet_to_bin(data: str) -> str:
    return "".join(HEX_TO_BIN[c] for c in data)


def get_packet_type(packet: str) -> PacketType:
    if packet == "100":
        return PacketType.Literal
    else:
        return PacketType.Operator


TPacket = Union[Packet, PacketLiteral]
TPacketTreeElement = Union[Packet, PacketLiteral]
TPacketTerminal = Union[PacketLiteral, PacketPad]


def parse_literal(
    data: str, header: PacketHeader, start: int, end: int | None = None
) -> tuple[TPacketTerminal, int | None]:
    idx = start
    first_bit = True
    is_last = False
    buffer = ""
    done_parsing = False
    current_read_count = 0

    while not done_parsing:
        if end is not None:
            if idx == end:
                raise ParseError("idx is beyond set end limit...")
        if first_bit is True:
            first_bit = False
            is_last = data[idx] == "0"
        else:
            buffer += data[idx]
            current_read_count += 1
            if current_read_count == 4:
                first_bit = True
                current_read_count = 0
                done_parsing = is_last
        idx += 1

    return (
        PacketLiteral(
            header=header,
            range=(start - 6, idx),
            value=buffer,
        ),
        idx,
    )


def parse(packet_data: str) -> Packet | PacketLiteral | tuple[PacketLiteral, PacketPad]:
    def parse_rec(
        data: str, start: int, end: int | None = None
    ) -> Packet | PacketLiteral:
        if end is not None:
            if start >= end:
                raise ValueError(f"Should not have read this far: {(start, end)=}")
        idx = start
        header = get_header(data, start)
        idx += 6  # advance index by header length
        if header.is_terminal:
            single_literal_packet, single_literal_packet_end = parse_literal(
                data=data, header=header, start=idx
            )
            return single_literal_packet

        else:  # header is not terminal
            children = []
            packet_length_type = PacketLengthTypeId(data[idx])
            idx += 1  # advance by length of length type id

            if packet_length_type == PacketLengthTypeId.TotalLengthOfSubPackets:
                chunk_size = 15

            elif packet_length_type == PacketLengthTypeId.NumberOfSubPacketsContained:
                chunk_size = 11
            else:
                raise ValueError(f"Cannot understand {packet_length_type}")

            number = int(data[idx: idx + chunk_size], 2)
            idx += chunk_size

            if packet_length_type == PacketLengthTypeId.TotalLengthOfSubPackets:
                max_length = idx + number

                while idx < max_length:
                    sub_packet = parse_rec(data=data, start=idx)
                    _, sp_end = sub_packet.range
                    if sp_end <= max_length:
                        children.append(sub_packet)
                        idx = sp_end
                    else:
                        break
            else:
                assert (
                    packet_length_type == PacketLengthTypeId.NumberOfSubPacketsContained
                )
                for _ in range(number):
                    sub_packet = parse_rec(data=data, start=idx)
                    children.append(sub_packet)
                    _, sp_end = sub_packet.range
                    idx = sp_end
            return Packet(
                header=header,
                length_type_id=packet_length_type,
                children=[*children],
                range=(start, idx),
            )

    top_packet = parse_rec(data=packet_data, start=0)

    return top_packet


TAggregatorResult = TypeVar("TAggregatorResult")

AggregatorFuncTypeDef = Callable[[Iterable[TPacket]], TAggregatorResult]
AggregatorFuncProviderTypeDef = Callable[[TPacket], AggregatorFuncTypeDef]


def solve(packet: Packet | PacketLiteral) -> int:
    def traverse_rec(node: Packet | PacketLiteral) -> int:
        if node.packet_type == PacketType.Literal:
            return int(node.value, 2)
        else:
            packet_packet_id = int(node.header.packet_type_id, 2)
            if packet_packet_id == 0:
                return sum(traverse_rec(child) for child in node.children)
            if packet_packet_id == 1:
                return reduce(
                    lambda acc, e: acc * e,
                    (traverse_rec(child) for child in node.children),
                )
            if packet_packet_id == 2:
                return min(traverse_rec(child) for child in node.children)
            if packet_packet_id == 3:
                return max(traverse_rec(child) for child in node.children)
            if packet_packet_id == 5:
                return int(
                    traverse_rec(node.children[0]) > traverse_rec(node.children[1])
                )
            if packet_packet_id == 6:
                return int(
                    traverse_rec(node.children[0]) < traverse_rec(node.children[1])
                )
            if packet_packet_id == 7:
                return int(
                    traverse_rec(node.children[0]) == traverse_rec(node.children[1])
                )

    return traverse_rec(packet)


def traverse(packet) -> list[TPacketTreeElement]:
    result: list[TPacketTreeElement] = []
    queue: list[TPacketTreeElement] = [packet]

    while queue:
        item = queue.pop()
        if item.packet_type == PacketType.Pad:
            continue
        result.append(item)
        if hasattr(item, "children"):
            for child in item.children:
                queue.append(child)

    return result


def compute(s: str) -> int:
    packet_in_bin = packet_to_bin(s)
    top = parse(packet_data=packet_in_bin)
    # return sum(int(p.header.packet_version, 2) for p in traverse(top))
    return solve(top)


INPUT_S = """\
D2FE28
"""


@pytest.mark.parametrize(
    ("input_s", "expected"),
    ((INPUT_S, 1),),
)
def test(input_s: str, expected: int) -> None:
    assert compute("620080001611562C8802118E34") == 12


@pytest.mark.parametrize(
    ("input_s", "expected"),
    (
        ("C200B40A82", 3),
        ("04005AC33890", 54),
        ("880086C3E88112", 7),
        ("CE00C43D881120", 9),
        ("D8005AC2A8F0", 1),
        ("F600BC2D8F", 0),
        ("9C005AC2F8F0", 0),
        ("9C0141080250320F1802104A08", 1),
    ),
)
def test_part_2(input_s, expected):
    packet = packet_to_bin(input_s)
    parsed = parse(packet)
    assert solve(parsed) == expected


def test_packet_to_bin():
    actual = packet_to_bin("38006F45291200")
    assert actual == "00111000000000000110111101000101001010010001001000000000"


def test_parse_packet_single_literal():
    packet = "110100101111111000101000"
    actual = parse(packet_data=packet)
    actual_packet = actual

    assert isinstance(actual_packet, PacketLiteral)

    number = int(actual_packet.value, 2)

    assert number == 2021
    assert actual_packet.header.packet_version == "110"
    assert actual_packet.header.packet_type_id == "100"
    assert actual_packet.header.packet_type == PacketType.Literal


def test_parse_packet_total_length():
    packet = "00111000000000000110111101000101001010010001001000000000"
    actual = parse(packet_data=packet)

    assert actual
    assert len(actual.children) == 2
    assert (
        len(
            [
                c.packet_type
                for c in actual.children
                if c.packet_type == PacketType.Literal
            ]
        )
        == 2
    )
    assert [int(c.value, 2) for c in actual.children] == [10, 20]


def test_parse_packet_number_of_packets():
    packet = "11101110000000001101010000001100100000100011000001100000"
    actual = parse(packet_data=packet)
    assert actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", nargs="?", default=INPUT_TXT)
    args = parser.parse_args()

    with open(args.data_file) as f, timing():
        print(compute(f.read().rstrip()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
