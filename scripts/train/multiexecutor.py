
from pyexpat import model
from unittest import loader

import torch

from scripts.train.single_executor import load_programs_csv
from scripts.simulator import execute_instruction
from src.model.multiexecutor import MultiStepExecutor
from torch.utils.data import DataLoader

class MultiStepDataset(torch.utils.data.Dataset):
    def __init__(self, programs):
        self.samples = []

        for initial_state, instructions in programs:
            expected_state = initial_state

            for instruction in instructions:
                expected_state = execute_instruction(
                    expected_state,
                    instruction,
                )

            opcodes = [
                int(i.opcode)
                for i in instructions
            ]

            arg1s = [
                i.arg1
                for i in instructions
            ]

            arg2s = [
                NONE_REGISTER
                if i.arg2 is None
                else i.arg2
                for i in instructions
            ]

            self.samples.append(
                (
                    torch.tensor(
                        initial_state,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        opcodes,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        arg1s,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        arg2s,
                        dtype=torch.long,
                    ),
                    torch.tensor(
                        expected_state,
                        dtype=torch.long,
                    ),
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

NONE_REGISTER = 4




def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_programs_csv("data/phase2/multi_train.csv")

    dataset = MultiStepDataset(data)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = MultiStepExecutor().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    for epoch in range(50):

        accuracy_per_length = {}
        model.train()
        total_loss = 0.0
        correct_states = 0
        total_examples = 0
        for batch in loader:

            (
                initial_state,
                opcodes,
                arg1s,
                arg2s,
                final_state,
            ) = batch

            initial_state = initial_state.to(device)
            opcodes = opcodes.to(device)
            arg1s = arg1s.to(device)
            arg2s = arg2s.to(device)
            final_state = final_state.to(device)

            optimizer.zero_grad()

            logits = model(
                initial_state,
                opcodes,
                arg1s,
                arg2s,
            )

            loss = criterion(
                logits.reshape(-1, 10),
                final_state.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            # -------------------------
            # Metrics
            # -------------------------

            total_loss += loss.item()

            prediction = logits.argmax(dim=-1)


            exact = (
                prediction == final_state
            ).all(dim=1)

            batch_size = initial_state.size(0)
            program_length = opcodes.size(1)
            
            correct_in_batch = exact.sum().item()
            
            stats = accuracy_per_length.setdefault(
                program_length,
                {"correct": 0, "total": 0},
            )
            
            stats["correct"] += correct_in_batch
            stats["total"] += batch_size
            
            correct_states += correct_in_batch
            total_examples += batch_size

        avg_loss = total_loss / len(loader)
        state_accuracy = correct_states / total_examples

        accuracy_by_length = {
            length: f"{stats['correct'] / stats['total']:.2%}"
            for length, stats in accuracy_per_length.items()
        }

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"State Acc: {state_accuracy:.2%} | "
            f"Accuracy by Length: {accuracy_by_length}"
        )


    

    
if __name__ == "__main__":
    train()