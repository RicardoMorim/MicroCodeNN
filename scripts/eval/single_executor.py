from src.domain.instruction import Instruction, Opcode
import torch 
from src.model.single_executor import SingleExecutor
from collections import defaultdict
from scripts.train.single_executor import (
    _create_add_sub_metrics,
    _print_add_sub_metrics,
    _stack_state_batch,
    _update_add_sub_metrics,
    load_csv,
)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_samples = 0
    correct_registers = 0
    correct_states = 0
    correct_by_opcode = defaultdict(int)
    total_by_opcode = defaultdict(int)
    add_sub_metrics = _create_add_sub_metrics()

    with torch.no_grad():
        for batch in data_loader:
            state, opcode, arg1, arg2, next_state = batch

            state = _stack_state_batch(state).to(device).long()
            opcode = opcode.to(device).long()
            arg1 = arg1.to(device).long()
            arg2 = arg2.to(device).long()
            next_state = _stack_state_batch(next_state).to(device).long()

            logits = model(state, opcode, arg1, arg2)
            predicted_next_states = torch.argmax(logits, dim=-1)

            # Update per-opcode accuracy
            for i in range(len(opcode)):
                op = int(opcode[i].item())
                is_correct = (predicted_next_states[i] == next_state[i]).all()

                total_by_opcode[op] += 1
                if is_correct:
                    correct_by_opcode[op] += 1
            
            
            total_samples += state.size(0)
            correct_registers += (predicted_next_states == next_state).sum().item()
            correct_states += (predicted_next_states == next_state).all(dim=1).sum().item()

            _update_add_sub_metrics(
                add_sub_metrics,
                state,
                opcode,
                arg1,
                arg2,
                predicted_next_states,
                next_state,
            )

    register_accuracy = correct_registers / (total_samples * 4)  # 4 registers per sample
    state_accuracy = correct_states / total_samples

    print(f"Register Accuracy: {register_accuracy:.4f}")
    print(f"State Accuracy: {state_accuracy:.4f}")
    _print_add_sub_metrics("Eval", add_sub_metrics)
    print("Per-Opcode Accuracy:")
    for op, correct in correct_by_opcode.items():
        total = total_by_opcode[op]
        accuracy = correct / total if total > 0 else 0.0
        print(f"Opcode {Opcode(op).name}: {accuracy:.4f} ({correct}/{total})")

    print("Total Samples:", total_samples)
    print("Total Correct Registers:", correct_registers)
    print("Total Correct States:", correct_states)
    print("Total by Opcode:", dict(total_by_opcode))
    print("Correct by Opcode:", dict(correct_by_opcode))
    print("Opcode Accuracy:", {Opcode(op).name: (correct_by_opcode[op] / total_by_opcode[op] if total_by_opcode[op] > 0 else 0.0) for op in total_by_opcode})
    


    return register_accuracy, state_accuracy


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleExecutor().to(device)
    model.load_state_dict(torch.load("checkpoints/phase1/single_instruction_model_epoch_33.pt", map_location=device))
    # Load your data_loader here
    data = load_csv("data/phase0/test.csv") 

    data_loader = torch.utils.data.DataLoader(data, batch_size=8192, shuffle=False, num_workers=8)

    register_accuracy, state_accuracy = evaluate_model(model, data_loader, device)
    print(f"Register Accuracy: {register_accuracy:.4f}")
    print(f"State Accuracy: {state_accuracy:.4f}")