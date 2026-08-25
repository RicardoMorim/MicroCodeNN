from scripts.simulator import execute_instruction
from src.domain.instruction import Instruction, Opcode
from src.model.single_executor import SingleExecutor
import torch
from collections import defaultdict


NONE_REGISTER = 4


def _create_add_sub_metrics():
    return {
        "ADD": {
            "no_carry": {"correct": 0, "total": 0},
            "with_carry": {"correct": 0, "total": 0},
        },
        "SUB": {
            "no_borrow": {"correct": 0, "total": 0},
            "with_borrow": {"correct": 0, "total": 0},
        },
    }


def _update_add_sub_metrics(
    metrics,
    state,
    opcode,
    arg1,
    arg2,
    predictions,
    next_state,
):
    def _accumulate(mask, op_name, raw_condition_name, raw_condition):
        if not bool(mask.any()):
            return

        selected_state = state[mask]
        selected_arg1 = arg1[mask]
        selected_arg2 = arg2[mask]
        selected_predictions = predictions[mask]
        selected_next_state = next_state[mask]

        row_indices = torch.arange(selected_state.size(0), device=state.device)

        left_value = selected_state[row_indices, selected_arg1]
        right_value = selected_state[row_indices, selected_arg2]

        raw_result = left_value + right_value if op_name == "ADD" else left_value - right_value

        condition_mask = raw_condition(raw_result)

        if not bool(condition_mask.any()):
            return

        target_prediction = selected_predictions[row_indices, selected_arg1]
        target_expected = selected_next_state[row_indices, selected_arg1]

        target_correct = target_prediction == target_expected

        total = int(condition_mask.sum().item())
        correct = int((target_correct & condition_mask).sum().item())

        metrics[op_name][raw_condition_name]["total"] += total
        metrics[op_name][raw_condition_name]["correct"] += correct

    add_mask = opcode == int(Opcode.ADD)
    _accumulate(add_mask, "ADD", "no_carry", lambda raw: raw < 10)
    _accumulate(add_mask, "ADD", "with_carry", lambda raw: raw >= 10)

    sub_mask = opcode == int(Opcode.SUB)
    _accumulate(sub_mask, "SUB", "no_borrow", lambda raw: raw >= 0)
    _accumulate(sub_mask, "SUB", "with_borrow", lambda raw: raw < 0)


def _print_add_sub_metrics(prefix, metrics):
    def _line(op_name, case_name, label):
        correct = metrics[op_name][case_name]["correct"]
        total = metrics[op_name][case_name]["total"]
        accuracy = (correct / total) if total > 0 else 0.0

        print(
            f"{prefix} {op_name} {label}: "
            f"{accuracy:.2%} "
            f"({correct}/{total})"
        )

    _line("ADD", "no_carry", "no-carry")
    _line("ADD", "with_carry", "with-carry")
    _line("SUB", "no_borrow", "no-borrow")
    _line("SUB", "with_borrow", "with-borrow")


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


def _stack_state_batch(state_batch):
    """Convert DataLoader's list of register tensors to ``[B, 4]``."""
    return torch.stack(tuple(state_batch), dim=1)


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


def load_programs_csv(file_path):
    """Load each CSV row as an initial state and instruction sequence."""
    programs = []

    with open(file_path, "r") as file:
        next(file, None)  # Skip the State;Instruction header.
        for line_number, line in enumerate(file, start=2):
            groups = line.strip().split(";")
            if not groups or not groups[0]:
                continue

            initial_state = _parse_state(groups[0], line_number)
            instructions = tuple(
                _parse_instruction(group, line_number)
                for group in groups[1:]
            )
            programs.append((initial_state, instructions))

    return programs


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
        "data/phase1/single_instruction_train.csv"
    )

    validation_data = load_csv(
        "data/phase1/single_instruction_eval.csv"
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=256,
        shuffle=True,
        num_workers=7,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=256,
        shuffle=False,
        num_workers=7,
    )


    for epoch in range(100):

        # --------------------
        # TRAIN
        # --------------------

        model.train()
        correct_by_opcode = defaultdict(int)
        total_by_opcode = defaultdict(int)
        train_loss = 0.0
        train_correct_registers = 0
        train_correct_states = 0
        train_examples = 0
        train_add_sub_metrics = _create_add_sub_metrics()

        for batch in train_loader:

            state, opcode, arg1, arg2, next_state = batch

            state = _stack_state_batch(state).to(device).long()
            opcode = opcode.to(device).long()
            arg1 = arg1.to(device).long()
            arg2 = arg2.to(device).long()
            next_state = _stack_state_batch(next_state).to(device).long()

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
            exact = (predictions == next_state).all(dim=1)

            for op, is_correct in zip(opcode.cpu(), exact.cpu()):
                op = int(op.item())

                total_by_opcode[op] += 1

                if bool(is_correct):
                    correct_by_opcode[op] += 1

            # Register-level accuracy
            train_correct_registers += (
                predictions == next_state
            ).sum().item()

            _update_add_sub_metrics(
                train_add_sub_metrics,
                state,
                opcode,
                arg1,
                arg2,
                predictions,
                next_state,
            )

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
        validation_add_sub_metrics = _create_add_sub_metrics()

        with torch.no_grad():

            for batch in validation_loader:

                state, opcode, arg1, arg2, next_state = batch

                state = _stack_state_batch(state).to(device).long()
                opcode = opcode.to(device).long()
                arg1 = arg1.to(device).long()
                arg2 = arg2.to(device).long()
                next_state = _stack_state_batch(next_state).to(device).long()

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

                _update_add_sub_metrics(
                    validation_add_sub_metrics,
                    state,
                    opcode,
                    arg1,
                    arg2,
                    predictions,
                    next_state,
                )

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


        for op in Opcode:
            correct = correct_by_opcode[int(op)]
            total = total_by_opcode[int(op)]

            accuracy = (correct / total) if total > 0 else 0.0

            print(
                f"{op.name}: "
                f"{accuracy:.2%} "
                f"({correct}/{total})"
    )

        _print_add_sub_metrics("Train", train_add_sub_metrics)
        _print_add_sub_metrics("Val", validation_add_sub_metrics)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Reg Acc: {train_register_accuracy:.2%} | "
            f"Train State Acc: {train_state_accuracy:.2%} | "
            f"Val Loss: {avg_validation_loss:.4f} | "
            f"Val Reg Acc: {validation_register_accuracy:.2%} | "
            f"Val State Acc: {validation_state_accuracy:.2%}"
        )
        

        with open("checkpoints/phase1/single_instruction_train_log.csv", "a") as log_file:
            log_file.write(
                f"{epoch + 1},"
                f"{avg_train_loss:.4f},"
                f"{train_register_accuracy:.4f},"
                f"{train_state_accuracy:.4f},"
                f"{avg_validation_loss:.4f},"
                f"{validation_register_accuracy:.4f},"
                f"{validation_state_accuracy:.4f}\n"
            )

        torch.save(
            model.state_dict(),
            f"checkpoints/phase1/single_instruction_model_epoch_{epoch + 1}.pt",
        )


if __name__ == "__main__":
    train()