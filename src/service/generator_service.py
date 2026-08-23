import random

from src.domain.instruction import Opcode

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



def random_state():
    return tuple(
        random.randint(0, 9)
        for _ in range(4) # Fixed size of 4 for the state for now
    )

def random_instructions():
    num_instructions = random.randint(3, 8)  # Random number of instructions between 3 and 8
    instructions = []
    for _ in range(num_instructions):
        instruction = random.choice(SIMPLE_OPCODES + COMPLEX_OPCODES)
        index1 = random.randint(0, 3)

        if instruction in SIMPLE_OPCODES:
            instructions.append((str(instruction), index1, None))
        elif instruction in COMPLEX_OPCODES:
            index2 = get_non_matching_index(index1)
            instructions.append((str(instruction), index1, index2))

    return instructions


def get_non_matching_index(index1):
    index2 = random.randint(0, 3)
    while index2 == index1:
        index2 = random.randint(0, 3)
    return index2

