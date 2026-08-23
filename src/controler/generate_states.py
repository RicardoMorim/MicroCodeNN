from service.generator_service import random_state
from service.generator_service import random_instruction

def generate_states(n):
    """
    Generates states using the generate_states_service.

    Returns:
        list: A list of generated states.
    """


    return [random_state() for _ in range(n)]

def generate_instructions(n):
    """
    Generates instructions using the generate_states_service.

    Returns:
        list: A list of generated instructions.
    """

    return [random_instruction() for _ in range(n)]



