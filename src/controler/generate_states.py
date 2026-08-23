from src.service.generator_service import random_instructions, random_state

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

    return [random_instructions()]

if __name__ == "__main__":
    n = 200  # Number of sets of states and instructions to generate
    sets = []
    for i in range(n):
        states = generate_state()
        instructions = generate_instructions()
        sets.append((states, instructions))

    # Save the generated sets to a file
    with open("validation.txt", "w") as f:
        for states, instructions in sets:
            f.write(f"States: {states}, Instructions: {instructions}\n")    
