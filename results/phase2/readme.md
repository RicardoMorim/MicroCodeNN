Results latent executor:

python -m scripts.multiexecutor.latent.executor
Instruction Accuracy: 0.9882
Program Accuracy: 0.9807
Opcode DEC: 0.9909 (454602/458767)
Opcode ADD: 0.9830 (449719/457513)
Opcode INC: 0.9911 (454511/458603)
Opcode SUB: 0.9844 (450504/457660)
Opcode COPY: 0.9918 (454673/458435)
Opcode SWAP: 0.9883 (452232/457604)



Results multi executor:

The model shows clear signs of performance saturation. While the loss continues to decrease throughout training, the overall state accuracy improves only marginally after approximately 25–30 epochs, stabilizing around 33–34%.

A strong dependency on instruction sequence length is also visible. Performance consistently decreases as the number of instructions increases:

| Instructions | Final Accuracy |
|---:|---:|
| 3 | ~55.7% |
| 4 | ~44.6% |
| 5 | ~34.6% |
| 6 | ~27.4% |
| 7 | ~21.9% |
| 8 | ~17.6% |

This suggests that the model handles short execution sequences reasonably well, but its ability to correctly maintain and update the state degrades as the task requires more sequential instructions.

The gap is significant: sequences with **3 instructions achieve more than 3× the accuracy of sequences with 8 instructions**.

Additionally, after the initial learning phase, further training produces diminishing returns. For example, between epochs 30 and 43 the overall accuracy only increases from approximately **32.7% to 33.7%**, despite the continued decrease in training loss.


