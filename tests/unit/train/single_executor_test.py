import torch
from unittest.mock import mock_open, patch

from src.domain.instruction import Opcode
from scripts.train.single_executor import (
    _create_add_sub_metrics,
    _update_add_sub_metrics,
    load_csv,
)


def test_load_csv_reads_state_and_each_instruction():
    file_content = (
        "State;Instruction\n"
        "1,2,3,4;0,0,None;5,1,3\n"
    )

    with patch("builtins.open", mock_open(read_data=file_content)):
        data = load_csv("train.csv")

    assert data == [
        ((1, 2, 3, 4), 0, 0, 4, (2, 2, 3, 4)),
        ((2, 2, 3, 4), 5, 1, 3, (2, 4, 3, 2)),
    ]


def test_update_add_sub_metrics_tracks_add_no_carry_and_with_carry():
    metrics = _create_add_sub_metrics()

    state = torch.tensor(
        [
            [2, 3, 0, 0],
            [7, 8, 0, 0],
        ],
        dtype=torch.long,
    )
    opcode = torch.tensor(
        [int(Opcode.ADD), int(Opcode.ADD)],
        dtype=torch.long,
    )
    arg1 = torch.tensor([0, 0], dtype=torch.long)
    arg2 = torch.tensor([1, 1], dtype=torch.long)

    next_state = torch.tensor(
        [
            [5, 3, 0, 0],  # 2 + 3 = 5 (no carry)
            [5, 8, 0, 0],  # 7 + 8 = 15 -> 5 (with carry)
        ],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [
            [5, 3, 0, 0],  # correct target register
            [4, 8, 0, 0],  # wrong target register
        ],
        dtype=torch.long,
    )

    _update_add_sub_metrics(
        metrics,
        state,
        opcode,
        arg1,
        arg2,
        predictions,
        next_state,
    )

    assert metrics["ADD"]["no_carry"] == {"correct": 1, "total": 1}
    assert metrics["ADD"]["with_carry"] == {"correct": 0, "total": 1}


def test_update_add_sub_metrics_tracks_sub_no_borrow_and_with_borrow():
    metrics = _create_add_sub_metrics()

    state = torch.tensor(
        [
            [7, 3, 0, 0],
            [2, 8, 0, 0],
        ],
        dtype=torch.long,
    )
    opcode = torch.tensor(
        [int(Opcode.SUB), int(Opcode.SUB)],
        dtype=torch.long,
    )
    arg1 = torch.tensor([0, 0], dtype=torch.long)
    arg2 = torch.tensor([1, 1], dtype=torch.long)

    next_state = torch.tensor(
        [
            [4, 3, 0, 0],  # 7 - 3 = 4 (no borrow)
            [4, 8, 0, 0],  # 2 - 8 = -6 -> 4 (with borrow)
        ],
        dtype=torch.long,
    )

    predictions = torch.tensor(
        [
            [4, 3, 0, 0],  # correct target register
            [4, 8, 0, 0],  # correct target register
        ],
        dtype=torch.long,
    )

    _update_add_sub_metrics(
        metrics,
        state,
        opcode,
        arg1,
        arg2,
        predictions,
        next_state,
    )

    assert metrics["SUB"]["no_borrow"] == {"correct": 1, "total": 1}
    assert metrics["SUB"]["with_borrow"] == {"correct": 1, "total": 1}
