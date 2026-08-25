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


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
if __name__ == "__main__":
    train()