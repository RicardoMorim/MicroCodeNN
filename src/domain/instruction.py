from dataclasses import dataclass
from enum import IntEnum


class Opcode(IntEnum):
    INC = 0
    DEC = 1
    ADD = 2
    SUB = 3
    COPY = 4
    SWAP = 5


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    arg1: int
    arg2: int | None = None