from src.domain.instruction import Instruction, Opcode
from scripts.simulator import execute_instruction, execute_program



def test_add():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.ADD,
        0,
        1,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (0, 7, 1, 9)


def test_copy():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.COPY,
        0,
        2,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (1, 7, 1, 9)

def test_swap():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.SWAP,
        0,
        3,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (9, 7, 1, 3)

def test_sub():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.SUB,
        0,
        1,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (6, 7, 1, 9)


def test_inc():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.INC,
        0,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (4, 7, 1, 9)

def test_dec():
    state = (3, 7, 1, 9)

    instruction = Instruction(
        Opcode.DEC,
        0,
    )

    output = execute_instruction(
        state,
        instruction,
    )

    assert output == (2, 7, 1, 9)


def test_execute_program():
    initial_state = (0, 0, 0, 0)
    program = [
        Instruction(Opcode.INC, 0),
        Instruction(Opcode.INC, 1),
        Instruction(Opcode.ADD, 2, 0),
        Instruction(Opcode.SWAP, 1, 2),
    ]

    final_state, trajectory = execute_program(
        initial_state,
        program,
    )

    assert final_state == (1, 1, 1, 0)
    assert trajectory == [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        (1, 1, 1, 0),
        (1, 1, 1, 0),
    ]


def test_swap_twice_restores_state():
    initial_state = (3, 7, 1, 9)
    program = [
        Instruction(Opcode.SWAP, 0, 3),
        Instruction(Opcode.SWAP, 0, 3),
    ]

    final_state, trajectory = execute_program(
        initial_state,
        program,
    )

    assert final_state == initial_state

# Test extremes 
# INC 9 -> 0
# DEC 0 -> 9
# 9 + 9 -> 8
# 2 - 8 -> 4
# All mod 10
def test_modulus_behavior():
    initial_state = (9, 0, 9, 2)
    program = [
        Instruction(Opcode.INC, 0),  # 9 -> 0
        Instruction(Opcode.DEC, 1),  # 0 -> 9
        Instruction(Opcode.ADD, 2, 2), # 9 + 9 -> 8
        Instruction(Opcode.SUB, 3, 2), # 2 - 8 -> 4
    ]

    final_state, trajectory = execute_program(
        initial_state,
        program,
    )

    assert final_state == (0, 9, 8, 4)

def test_copy_and_swap():
    initial_state = (5, 3, 2, 1)
    program = [
        Instruction(Opcode.COPY, 0, 1),  # Copy value from index 1 to index 0
        Instruction(Opcode.SWAP, 2, 3),   # Swap values at index 2 and index 3
    ]

    final_state, trajectory = execute_program(
        initial_state,
        program,
    )

    assert final_state == (3, 3, 1, 2)

def test_multiple_operations():
    initial_state = (1, 2, 3, 4)
    program = [
        Instruction(Opcode.INC, 0),      # 1 -> 2
        Instruction(Opcode.ADD, 1, 0),    # 2 + 2 -> 4
        Instruction(Opcode.SUB, 2, 1),    # 3 - 4 -> 9 (modulus)
        Instruction(Opcode.DEC, 3),       # 4 -> 3
        Instruction(Opcode.SWAP, 0, 3),   # Swap index 0 and index 3
    ]

    final_state, trajectory = execute_program(
        initial_state,
        program,
    )

    assert final_state == (3, 4, 9, 2)

def test_unsupported_opcode():
    initial_state = (1, 2, 3, 4)
    program = [
        Instruction(Opcode.INC, 0),
        Instruction(Opcode.DEC, 1),
        Instruction(Opcode.ADD, 2, 3),
        Instruction(Opcode.SUB, 0, 1),
        Instruction(Opcode.COPY, 3, 2),
        Instruction(Opcode.SWAP, 1, 2),
        # Unsupported opcode
        Instruction("INVALID_OPCODE", 0),
    ]

    try:
        execute_program(
            initial_state,
            program,
        )
    except ValueError as e:
        assert str(e) == "Unsupported opcode: INVALID_OPCODE"

