# TinyLLMIdea — Neural Microcode & Compositional Execution Research

> Experimental research project exploring whether small neural networks can learn reusable computational operations, compose them over multiple steps, and eventually use those operations as an internal reasoning mechanism.

## Overview

Modern language models typically perform computation implicitly inside large Transformer networks.

This project explores a different question:

> Can a small neural system learn reusable computational primitives and compose them like instructions in a program?

The long-term idea is to investigate an architecture where a model does not have to perform all reasoning inside one undifferentiated neural transformation.

Instead, it could learn:

1. a representation of the current problem state;
2. reusable neural operations;
3. a mechanism for selecting and composing those operations;
4. an internal execution process;
5. eventually, a mapping between natural-language problems and these internal computations.

Conceptually:

```text
Problem
   ↓
Language / State Encoder
   ↓
Internal State
   ↓
Planner / Controller
   ↓
Neural Operation
   ↓
Updated Internal State
   ↓
Neural Operation
   ↓
...
   ↓
Answer
````

The project is intentionally developed incrementally.

Before attempting language, planning, routing, or learned programs, each lower-level capability is tested independently.

---

# Research Question

The main long-term research question is:

> Can a neural network learn a small set of reusable computational primitives, compose them into programs, and generalize those compositions beyond what was observed during training?

This can be decomposed into progressively harder questions:

```text
1. Can a neural network learn individual operations?

2. Can one shared neural transition execute multiple operations?

3. Can a continuous neural state preserve computation across multiple steps?

4. Does neural execution generalize to programs longer than those seen during training?

5. Are multiple specialized neural operators better than one shared transition?

6. Can a model learn which neural operator to use without being explicitly told?

7. Can a model infer or generate its own internal program?

8. Can natural-language problems be mapped into this execution mechanism?
```

The project is currently progressing through these questions one at a time.

---

# Why Start With a Tiny Virtual Machine?

The initial environment is deliberately artificial.

The goal is **not** to build a useful calculator or replace normal algorithms with neural networks.

A deterministic TinyVM gives us something extremely valuable for research:

```text
known input state
+
known instruction
+
known intermediate states
+
known final state
```

Because the true state of the virtual machine is known after every instruction, we can later compare the model's internal neural state against the actual computational state.

This makes it possible to distinguish:

```text
"The model returned the right answer"
```

from the much stronger claim:

```text
"The model internally maintained the correct computation across the program."
```

The TinyVM is therefore a controlled experimental environment for studying neural computation.

---

# TinyVM

The machine currently has four registers:

```text
R0
R1
R2
R3
```

Each register stores one value in:

```text
0 ... 9
```

Arithmetic is performed modulo 10.

For example:

```text
9 + 9 = 18
18 mod 10 = 8
```

and:

```text
2 - 8 = -6
-6 mod 10 = 4
```

The current instruction set is:

| Instruction | Semantics                     |
| ----------- | ----------------------------- |
| `INC A`     | `R[A] = (R[A] + 1) mod 10`    |
| `DEC A`     | `R[A] = (R[A] - 1) mod 10`    |
| `ADD A B`   | `R[A] = (R[A] + R[B]) mod 10` |
| `SUB A B`   | `R[A] = (R[A] - R[B]) mod 10` |
| `COPY A B`  | `R[A] = R[B]`                 |
| `SWAP A B`  | swap `R[A]` and `R[B]`        |

Example:

```text
Initial state:
(0, 0, 0, 0)

INC R0
→ (1, 0, 0, 0)

INC R1
→ (1, 1, 0, 0)

ADD R2 R0
→ (1, 1, 1, 0)
```

The Python simulator implements the real instruction semantics and is used only to generate ground truth.

The neural networks do **not** contain implementations such as:

```python
if opcode == ADD:
    state[a] = state[a] + state[b]
```

They must learn the transformations from data.

---

# Phase 0 — Initial Single-Step Executor

The first experiment asked the simplest possible question:

> Can a small neural network learn the transition
> `current_state + instruction -> next_state`?

The model received:

```text
4 register values
+
opcode
+
arg1
+
arg2
```

Each categorical input was mapped through a learned embedding.

## Model

```text
value embedding:    10 × 16
opcode embedding:    6 × 16
register embedding:  5 × 16

Input:
4 value embeddings       = 4 × 16 = 64
opcode embedding         = 16
arg1 embedding           = 16
arg2 embedding           = 16

Total input dimension    = 112
```

The MLP was:

```text
112
 ↓
128
 ↓ GELU
128
 ↓ GELU
40
```

The 40 outputs correspond to:

```text
4 registers × 10 possible values
```

and are reshaped into:

```text
[B, 4, 10]
```

Cross-entropy loss is applied independently to the four register predictions.

Optimizer:

```text
AdamW
lr = 1e-3
weight_decay = 1e-5
```

---

# Phase 0 Results

Training data:

```text
20,000 samples
```

Evaluation:

```text
5,000,000 samples
```

Results:

```text
Register Accuracy: 93.87%
Exact State Accuracy: 75.56%
```

Per-opcode exact-state accuracy:

| Opcode | Accuracy |
| ------ | -------: |
| INC    |   98.78% |
| DEC    |   98.15% |
| COPY   |   94.77% |
| SWAP   |   93.17% |
| ADD    |   34.04% |
| SUB    |   34.45% |

The important observation was that `ADD` and `SUB` were dramatically harder than the other instructions.

At this stage several explanations were still possible:

```text
insufficient data
poor numerical representation
model capacity
optimization difficulty
loss imbalance
architectural limitation
```

No architectural changes were made before testing the simplest explanation: insufficient training coverage.

---

# Phase 1 — Increased Data Coverage

The same model architecture was trained again using a much larger dataset.

Training data:

```text
500,000 samples
```

No major architecture or loss changes were required.

## Results

```text
Register Accuracy: 99.91%
Exact State Accuracy: 99.63%
```

Per-opcode exact-state accuracy:

| Opcode | Accuracy |
| ------ | -------: |
| INC    |   99.96% |
| DEC    |   99.95% |
| COPY   |   99.83% |
| SWAP   |   99.66% |
| ADD    |   99.12% |
| SUB    |   99.27% |

Arithmetic-specific evaluation:

```text
ADD no-carry:     99.08%
ADD with-carry:   99.28%

SUB no-borrow:    99.45%
SUB with-borrow:  99.15%
```

This substantially changed the interpretation of Phase 0.

The weak `ADD` and `SUB` performance was not evidence of a hard architectural limitation.

Increasing data coverage was enough for the same small model to learn all six one-step transitions with very high accuracy.

---

# Phase 1 Conclusion

The main conclusion is:

> One-step instruction execution is no longer the dominant bottleneck.

The experiment demonstrates that the current model has enough capacity to approximate the TinyVM's atomic transition function with very high accuracy.

It does **not** yet demonstrate reasoning, algorithmic generalization, or compositional extrapolation.

The one-step state space is small, and the large training dataset may cover a substantial fraction of possible transitions.

Therefore Phase 1 should be interpreted as a capability / sanity check:

```text
Can the model learn the atomic machine?

Yes, to approximately 99%+ accuracy.
```

The next question is much more important:

```text
Can those learned transformations be composed?
```

---

# Phase 2 — Multi-Step Neural Execution

Phase 2 moves from:

```text
state + one instruction
        ↓
next state
```

to:

```text
initial state
    ↓
instruction 1
    ↓
instruction 2
    ↓
instruction 3
    ↓
...
    ↓
final state
```

Two forms of execution are important.

## Discrete Rollout Baseline

The Phase 1 model can simply be called repeatedly:

```text
S0
 ↓ model(I1)
Ŝ1
 ↓ argmax
 ↓ model(I2)
Ŝ2
 ↓ argmax
 ↓ model(I3)
Ŝ3
```

This measures how errors accumulate when the atomic executor is chained algorithmically.

It provides an important baseline.

---

# Continuous Latent Executor

The more interesting Phase 2 model avoids converting the internal state back to discrete register values after every instruction.

Instead:

```text
S0
 ↓
Encoder
 ↓
H0

H0 + I1
 ↓
Transition
 ↓
H1

H1 + I2
 ↓
Transition
 ↓
H2

H2 + I3
 ↓
Transition
 ↓
H3

...
 ↓
Decoder
 ↓
Final State
```

Mathematically:

```text
H0 = Encoder(S0)

H(t+1) = H(t) + F(H(t), I(t))

Prediction = Decoder(H(T))
```

where:

```text
H(t)
```

is a continuous hidden representation of the current computational state.

No `argmax` or discrete state conversion occurs between instructions.

---

# Phase 2 Model

The current proposed model uses:

```text
embedding_dim = 16
hidden_dim = 128
```

## State Encoder

Each of the four values becomes a 16-dimensional embedding:

```text
[B, 4]
 ↓
[B, 4, 16]
 ↓ flatten
[B, 64]
```

The encoder maps:

```text
64 → 128
```

producing:

```text
H0 ∈ R^128
```

---

## Instruction Representation

Each instruction contains:

```text
opcode
arg1
arg2
```

Each receives a 16-dimensional embedding:

```text
16 + 16 + 16 = 48
```

Therefore:

```text
I(t) ∈ R^48
```

---

## Neural Transition

At each step:

```text
hidden state       128
instruction         48
-----------------------
total              176
```

The transition network is:

```text
176
 ↓
128
 ↓ GELU
128
```

It produces a change:

```text
ΔH(t)
```

and the hidden state is updated residually:

```text
H(t+1) = H(t) + ΔH(t)
```

The same transition network is reused for every program step.

This is the central mechanism being tested.

---

## Decoder

After the final instruction:

```text
H(T): 128
```

is mapped to:

```text
40 logits
```

and reshaped into:

```text
[B, 4, 10]
```

corresponding to the ten possible values of each of the four registers.

---

# Why a Continuous Hidden State?

A discrete rollout forces the network to commit to a value after every instruction:

```text
neural state
 ↓
argmax
 ↓
0...9
```

Information that does not survive this discrete conversion is lost.

A continuous executor instead allows the network to maintain an internal representation across the entire program.

This also allows gradients from the final result to propagate through every previous execution step during training.

The experiment therefore asks:

> Can a reusable neural transition maintain a computational state over multiple sequential operations?

---

# Phase 2 Experimental Plan

The initial experiments should remain simple.

Start with programs of exactly:

```text
length = 2
```

Once the model learns those successfully, increase progressively:

```text
2
3
4
5
6
```

The main training range will eventually be:

```text
1–6 instructions
```

Then the frozen model will be evaluated on longer programs such as:

```text
8
10
12
16
20
```

The primary graph will be:

```text
Exact Final-State Accuracy
          vs
Program Length
```

This tests length extrapolation directly.

---

# Main Phase 2 Comparison

The first important comparison will be:

| Model               | Description                                       |
| ------------------- | ------------------------------------------------- |
| Discrete Rollout    | Phase 1 executor repeatedly decoded with `argmax` |
| Continuous Executor | Persistent 128-dimensional latent state           |

If the continuous model retains substantially higher accuracy as program length increases, that would suggest that a learned continuous computational state improves neural composition.

If both behave similarly, the continuous state may provide little advantage.

If the continuous executor performs worse, that is also informative.

---

# Phase 3 — Neural Microcode

Phase 2 deliberately uses only one shared transition:

```text
F
```

for every instruction:

```text
INC  → F
DEC  → F
ADD  → F
SUB  → F
COPY → F
SWAP → F
```

The next major hypothesis is that one generic transition may not be the best way to represent qualitatively different computational operations.

Phase 3 will therefore introduce multiple learned neural operators:

```text
F0
F1
F2
F3
F4
F5
```

Initially an oracle can route each opcode to a corresponding learned operator.

Later, routing itself can be learned.

The key experiment becomes:

> Do specialized reusable neural primitives extrapolate to longer compositions better than a single shared neural transition?

This is the core **Neural Microcode** hypothesis.

---

# Long-Term Direction

If neural operators demonstrate meaningful compositional generalization, later phases could progressively remove the explicit structure provided by TinyVM.

Possible progression:

```text
Known opcode
    ↓
Learned operator routing
    ↓
Unsupervised operator specialization
    ↓
Unknown / arbitrary instruction symbols
    ↓
Instruction induction
    ↓
Learned program counter
    ↓
Learned planner
    ↓
Natural-language problem
    ↓
Internal neural program
    ↓
Answer
```

The eventual architecture could resemble:

```text
Natural Language
      ↓
Transformer Encoder
      ↓
Planner
      ↓
Latent Instruction
      ↓
Router
   ┌──┼──┐
   ↓  ↓  ↓
  OP0 OP1 OP2 ...
   └──┼──┘
      ↓
Working State
      ↓
next internal operation
      ↓
...
      ↓
HALT
      ↓
Language Decoder
      ↓
Answer
```

At that stage the model would no longer be calling external symbolic algorithms.

The operations themselves would be learned neural transformations.

---

# Experimental Philosophy

The project follows a deliberately incremental methodology.

Each phase should answer one question before adding another mechanism.

In particular:

```text
Do not add routing before proving transitions work.

Do not add multiple operators before measuring the single-transition baseline.

Do not add planning before demonstrating compositional execution.

Do not add natural language before the underlying execution mechanism works.
```

This makes negative results interpretable.

If a complex model fails, the project should be able to identify which capability failed rather than having several new mechanisms changing simultaneously.

---

# Reproducibility

Important experiments should record:

```text
Git commit / source hash
dataset generator configuration
dataset seed
training sample count
evaluation sample count
model parameters
optimizer
learning rate
weight decay
batch size
number of optimizer steps / epochs
checkpoint hash
device
precision
training time
evaluation metrics
```

Training and evaluation should also explicitly check for dataset overlap when generalization claims depend on unseen samples.

---

# Current Status

```text
Phase 0 ✅
Small-data atomic instruction learning.

Phase 1 ✅
High-coverage atomic instruction learning.
~99.6% exact-state accuracy.

Phase 2 🚧
Multi-step composition and continuous latent execution.

Phase 3 ⏳
Specialized neural micro-operators.

Phase 4+ ⏳
Learned routing, program induction, planning, and natural language.
```

---

# Current Main Hypothesis

The next hypothesis being tested is:

> A continuous neural state updated repeatedly by a learned transition can execute multi-step programs and may generalize to longer compositions better than repeatedly discretizing the state after every instruction.

If that succeeds, the following hypothesis will be:

> Multiple specialized neural operators can provide better compositional generalization than a single shared transition function.

The project should be considered successful only when improvements survive controlled baselines, unseen compositions, multiple random seeds, and appropriate compute/parameter comparisons.

Negative results are equally useful: the goal is to determine whether the proposed computational mechanisms actually work, not to force a positive conclusion.

