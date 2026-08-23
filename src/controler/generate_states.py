from src.service.generator_service import random_instructions, random_state


def format_training_example(states, instructions):
    """Format a state and its instructions as a semicolon-delimited row."""
    state_values = ",".join(str(value) for value in states)
    instruction_values = ";".join(
        ",".join(str(value) for value in instruction)
        for instruction in instructions
    )

    return f"{state_values};{instruction_values}"


def generate_state():
    """
    Generates states using the generate_states_service.

    Returns:
        list: A list of generated states.
    """

    return random_state()

def generate_instructions():
    """
    Generates instructions using the generate_states_service.

    Returns:
        list: A list of generated instructions.
    """

    return random_instructions()

if __name__ == "__main__":
    n = 50000  # Number of sets of states and instructions to generate
    sets = []
    for i in range(n):
        states = generate_state()
        instructions = generate_instructions()
        sets.append((states, instructions))

    # Save the generated sets to a semicolon-delimited file.
    with open("data/phase0/single_instruction_train.csv", "w") as f:
        f.write("State;Instruction\n")
        for states, instructions in sets:
            f.write(f"{format_training_example(states, instructions)}\n")
