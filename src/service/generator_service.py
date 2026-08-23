import random

from domain.instruction import Opcode

SIMPLE_OPCODES = [
    Opcode.INC,
    Opcode.DEC,
]

COMPLEX_OPCODES = [
    Opcode.COPY,
    Opcode.SWAP,
    Opcode.SUB,
    Opcode.ADD,
]

random.seed(42)



def random_state(rng):
    return tuple(
        rng.randrange(10)
        for _ in range(4) # Fixed size of 4 for the state for now
    )

def random_instruction():
    instruction = random.choice(SIMPLE_OPCODES + COMPLEX_OPCODES)

    index1 = random.randint(0, 3)

    if instruction in SIMPLE_OPCODES:
        return (instruction,index1, None)

    if instruction in COMPLEX_OPCODES:
        index2 = get_non_matching_index(index1) 
        return (instruction, index1, index2)



def get_non_matching_index(index1):
    index2 = random.randint(0, 3)
    while index2 == index1:
        index2 = random.randint(0, 3)
    return index2

