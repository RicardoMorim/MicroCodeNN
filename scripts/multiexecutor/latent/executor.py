from collections import defaultdict

import torch

from scripts.train.single_executor import load_programs_csv
from scripts.simulator import execute_instruction
from src.model.single_executor import SingleExecutor
from src.domain.instruction import Opcode


NONE_REGISTER = 4


def _predict_next_state(model, state, instruction, device):
    state_tensor = torch.tensor(
        state,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    opcode = torch.tensor(
        [int(instruction.opcode)],
        dtype=torch.long,
        device=device,
    )
    arg1 = torch.tensor([instruction.arg1], dtype=torch.long, device=device)
    arg2 = torch.tensor(
        [NONE_REGISTER if instruction.arg2 is None else instruction.arg2],
        dtype=torch.long,
        device=device,
    )

    logits = model(state_tensor, opcode, arg1, arg2)
    return tuple(torch.argmax(logits, dim=-1)[0].tolist())


def eval(model, programs, device):
    """Run every program sequentially using the phase1 single-step model."""
    model.eval()
    total_instructions = 0
    correct_instructions = 0
    total_programs = 0
    correct_programs = 0
    correct_by_opcode = defaultdict(int)
    total_by_opcode = defaultdict(int)

    with torch.no_grad():
        for initial_state, instructions in programs:
            predicted_state = initial_state
            expected_state = initial_state
            program_is_correct = True
            total_programs += 1

            for instruction in instructions:
                expected_state = execute_instruction(expected_state, instruction)
                predicted_state = _predict_next_state(
                    model,
                    predicted_state,
                    instruction,
                    device,
                )

                opcode = int(instruction.opcode)
                is_correct = predicted_state == expected_state
                total_instructions += 1
                total_by_opcode[opcode] += 1

                if is_correct:
                    correct_instructions += 1
                    correct_by_opcode[opcode] += 1
                else:
                    program_is_correct = False

            if program_is_correct:
                correct_programs += 1

    instruction_accuracy = (
        correct_instructions / total_instructions
        if total_instructions
        else 0.0
    )
    program_accuracy = correct_programs / total_programs if total_programs else 0.0

    print(f"Instruction Accuracy: {instruction_accuracy:.4f}")
    print(f"Program Accuracy: {program_accuracy:.4f}")
    for opcode, total in total_by_opcode.items():
        print(
            f"Opcode {Opcode(opcode).name}: "
            f"{correct_by_opcode[opcode] / total:.4f} "
            f"({correct_by_opcode[opcode]}/{total})"
        )

    return instruction_accuracy, program_accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleExecutor().to(device)
    model.load_state_dict(torch.load(
        "checkpoints/phase1/single_instruction_model_epoch_33.pt",
        map_location=device,
    ))
    programs = load_programs_csv("data/phase2/multi_eval.csv")
    eval(model, programs, device)