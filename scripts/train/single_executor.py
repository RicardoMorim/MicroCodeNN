from scripts.simulator import execute_instruction
from src.domain.instruction import Instruction, Opcode
from src.model.single_executor import SingleExecutor
import torch

NONE_REGISTER = 4


def _parse_state(raw_state, line_number):
    state = tuple(int(value) for value in raw_state.split(","))
    if len(state) != 4:
        raise ValueError(f"Invalid state on line {line_number}")
    return state


def _parse_instruction(raw_instruction, line_number):
    values = raw_instruction.split(",")
    if len(values) != 3:
        raise ValueError(f"Invalid instruction on line {line_number}")

    opcode, arg1, raw_arg2 = values
    return Instruction(
        Opcode(int(opcode)),
        int(arg1),
        None if raw_arg2 == "None" else int(raw_arg2),
    )


def _append_instruction_sample(data, state, instruction):
    next_state = execute_instruction(state, instruction)
    model_arg2 = (
        NONE_REGISTER if instruction.arg2 is None else instruction.arg2
    )
    data.append(
        (
            state,
            int(instruction.opcode),
            instruction.arg1,
            model_arg2,
            next_state,
        )
    )
    return next_state


def load_csv(file_path):
    """Load semicolon-delimited programs as single-step training samples.

    Each input row has the following format::

        state_0,state_1,state_2,state_3;opcode,arg1,arg2;...

    A sample is created for every instruction. Its target is the state after
    executing that instruction, preserving the return shape used by training.
    """
    data = []

    with open(file_path, "r") as file:
        next(file, None)  # Skip the State;Instruction header.
        for line_number, line in enumerate(file, start=2):
            groups = line.strip().split(";")
            if not groups or not groups[0]:
                continue

            current_state = _parse_state(groups[0], line_number)
            for group in groups[1:]:
                instruction = _parse_instruction(group, line_number)
                current_state = _append_instruction_sample(
                    data,
                    current_state,
                    instruction,
                )

    return data


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SingleExecutor().to(device)

    print(model)
    print("Device:", device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    train_data = load_csv(
        "data/phase0/single_instruction_train.csv"
    )

    validation_data = load_csv(
        "data/phase0/single_instruction_validation.csv"
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1024,
        shuffle=True,
        num_workers=7,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=1024,
        shuffle=False,
        num_workers=7,
    )

    for epoch in range(10):

        # --------------------
        # TRAIN
        # --------------------

        model.train()

        train_loss = 0.0
        train_correct_registers = 0
        train_correct_states = 0
        train_examples = 0

        for batch in train_loader:

            state, opcode, arg1, arg2, next_state = batch

            state = state.to(device).long()
            opcode = opcode.to(device).long()
            arg1 = arg1.to(device).long()
            arg2 = arg2.to(device).long()
            next_state = next_state.to(device).long()

            optimizer.zero_grad()

            logits = model(
                state,
                opcode,
                arg1,
                arg2,
            )

            # logits: [B, 4, 10]
            # next_state: [B, 4]

            loss = loss_fn(
                logits.reshape(-1, 10),
                next_state.reshape(-1),
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            predictions = logits.argmax(dim=-1)

            # Register-level accuracy
            train_correct_registers += (
                predictions == next_state
            ).sum().item()

            # Exact-state accuracy
            exact = (
                predictions == next_state
            ).all(dim=1)

            train_correct_states += exact.sum().item()
            train_examples += state.size(0)

        # --------------------
        # VALIDATION
        # --------------------

        model.eval()

        validation_loss = 0.0
        validation_correct_registers = 0
        validation_correct_states = 0
        validation_examples = 0

        with torch.no_grad():

            for batch in validation_loader:

                state, opcode, arg1, arg2, next_state = batch

                state = state.to(device).long()
                opcode = opcode.to(device).long()
                arg1 = arg1.to(device).long()
                arg2 = arg2.to(device).long()
                next_state = next_state.to(device).long()

                logits = model(
                    state,
                    opcode,
                    arg1,
                    arg2,
                )

                loss = loss_fn(
                    logits.reshape(-1, 10),
                    next_state.reshape(-1),
                )

                validation_loss += loss.item()

                predictions = logits.argmax(dim=-1)

                validation_correct_registers += (
                    predictions == next_state
                ).sum().item()

                exact = (
                    predictions == next_state
                ).all(dim=1)

                validation_correct_states += exact.sum().item()
                validation_examples += state.size(0)

        avg_train_loss = train_loss / len(train_loader)

        avg_validation_loss = (
            validation_loss
            / len(validation_loader)
        )

        train_register_accuracy = (
            train_correct_registers
            / (train_examples * 4)
        )

        train_state_accuracy = (
            train_correct_states
            / train_examples
        )

        validation_register_accuracy = (
            validation_correct_registers
            / (validation_examples * 4)
        )

        validation_state_accuracy = (
            validation_correct_states
            / validation_examples
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Reg Acc: {train_register_accuracy:.2%} | "
            f"Train State Acc: {train_state_accuracy:.2%} | "
            f"Val Loss: {avg_validation_loss:.4f} | "
            f"Val Reg Acc: {validation_register_accuracy:.2%} | "
            f"Val State Acc: {validation_state_accuracy:.2%}"
        )