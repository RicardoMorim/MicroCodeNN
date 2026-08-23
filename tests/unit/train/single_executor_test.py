from unittest.mock import mock_open, patch

from scripts.train.single_executor import load_csv


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
