from typing import Sequence

from src.domain.instruction import Instruction, Opcode


MODULUS = 10
NUM_REGISTERS = 4


def execute_instruction(
    state: Sequence[int],
    instruction: Instruction,
) -> tuple[int, ...]:

    result = list(state)

    a = instruction.arg1
    b = instruction.arg2

    if instruction.opcode == Opcode.INC:
        result[a] = (result[a] + 1) % MODULUS

    elif instruction.opcode == Opcode.DEC:
        result[a] = (result[a] - 1) % MODULUS

    elif instruction.opcode == Opcode.ADD:
        assert b is not None
        result[a] = (result[a] + result[b]) % MODULUS

    elif instruction.opcode == Opcode.SUB:
        assert b is not None
        result[a] = (result[a] - result[b]) % MODULUS

    elif instruction.opcode == Opcode.COPY:
        assert b is not None
        result[a] = result[b]

    elif instruction.opcode == Opcode.SWAP:
        assert b is not None
        result[a], result[b] = result[b], result[a]

    else:
        raise ValueError(
            f"Unsupported opcode: {instruction.opcode}"
        )

    return tuple(result)

def execute_program(
    initial_state,
    program,
):
    state = tuple(initial_state)

    trajectory = [state]

    for instruction in program:
        state = execute_instruction(
            state,
            instruction,
        )

        trajectory.append(state)

    return state, trajectory


if __name__ == "__main__":

    initial_state = (0, 0, 0, 0)
    program = [
        Instruction(Opcode.INC, 0),
        Instruction(Opcode.INC, 1),
        Instruction(Opcode.ADD, 2, 0),
        Instruction(Opcode.SWAP, 1, 2),
    ]

    final_state, trajectory = execute_program(initial_state, program)

    print("Final state:", final_state)
    print("Trajectory:")
    for state in trajectory:
        print(state)