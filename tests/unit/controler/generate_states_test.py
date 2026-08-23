from src.controler.generate_states import (
    format_training_example,
    generate_instructions,
)


def test_format_training_example_uses_requested_delimiters():
    states = (6, 5, 4, 2)
    instructions = [("1", 2, None)]

    output = format_training_example(states, instructions)

    assert output == "6,5,4,2;1,2,None"


def test_format_training_example_separates_multiple_instructions():
    states = (0, 1, 2, 3)
    instructions = [("0", 1, None), ("5", 2, 3)]

    output = format_training_example(states, instructions)

    assert output == "0,1,2,3;0,1,None;5,2,3"


def test_generate_instructions_returns_instruction_groups_directly():
    instructions = generate_instructions()

    assert instructions
    assert all(isinstance(instruction, tuple) for instruction in instructions)
