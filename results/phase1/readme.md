# Phase 1 Summary

After Phase 0 produced poor results for `ADD` and `SUB`, I decided to generate a larger and richer dataset.

That dataset was sufficient to make the model learn nearly everything with strong accuracy, including the previously hardest operations. All opcodes reached above 0.9 accuracy with a small model trained on 500k samples.

The evaluation was done on the same 5M dataset used in Phase 0.

## Results

- Register Accuracy: **0.9991**
- State Accuracy: **0.9963**
- Eval ADD no-carry: **99.08%** (453,501 / 457,696)
- Eval ADD with-carry: **99.28%** (372,351 / 375,054)
- Eval SUB no-borrow: **99.45%** (455,688 / 458,212)
- Eval SUB with-borrow: **99.15%** (372,134 / 375,330)

### Per-Opcode Accuracy

- DEC: **0.9995** (832,033 / 832,429)
- SWAP: **0.9966** (829,797 / 832,602)
- SUB: **0.9927** (827,442 / 833,542)
- INC: **0.9996** (834,435 / 834,774)
- COPY: **0.9983** (832,492 / 833,903)
- ADD: **0.9912** (825,418 / 832,750)

### Overall Stats

- Total Samples: **5,000,000**
- Total Correct Registers: **19,981,617**
- Total Correct States: **4,981,617**
- Total by Opcode: `{1: 832429, 5: 832602, 3: 833542, 0: 834774, 4: 833903, 2: 832750}`
- Correct by Opcode: `{1: 832033, 5: 829797, 3: 827442, 0: 834435, 4: 832492, 2: 825418}`
- Opcode Accuracy: `{'DEC': 0.9995242837527285, 'SWAP': 0.9966310434036911, 'SUB': 0.9926818324691498, 'INC': 0.9995939020621151, 'COPY': 0.998307956680813, 'ADD': 0.991195436805764}`

## Interpretation

> The model is clearly able to learn the instructions themselves at a very high level of accuracy.

We could try making the model larger, using different embeddings, or tuning the architecture to squeeze out even better results. However, the current model can approximate the one-step transition function for all six instructions with very high accuracy under this data regime.

The real challenge now is not instruction learning itself, but making the model reason about those instructions and execute them across multiple steps. That is the focus of the next phase.

## Next Steps

1. Plan and implement Phase 2, focused on teaching the model to reason about instructions and execute them step by step.
2. Compare two approaches:
   - letting the model perform the reasoning internally and output only the final result for multi-step tasks;
   - using the Phase 1 model inside an algorithmic loop where it sees the current state and the next instruction to execute, outputs the next state, and repeats until all instructions are complete.
3. If there is not a large difference in accuracy, we can transfer this knowledge into an LLM by training it to think in a more algorithmic way.



