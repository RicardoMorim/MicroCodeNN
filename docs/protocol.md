# μCode Experiment v0

## Research question

Can learned neural computational primitives be composed into
programs longer than those observed during training?

## Main hypothesis

A modular neural executor with specialized operators will
generalize better to longer program lengths than:

1. a single shared neural executor;
2. a direct Transformer baseline.

## Train distribution

Program length: 1-6

## Development extrapolation

Program length:
7, 8, 10, 12

## Final holdout

Program length:
14, 16, 18, 20

The final holdout must not be inspected during development.

## Operations

INC
DEC
ADD
SUB
COPY
SWAP

## Registers

R0-R3

## Values

0-9

All arithmetic modulo 10.

## Primary metric

Exact final-state accuracy.

A prediction is correct only if all four register values are correct.

## Secondary metrics

- per-register accuracy
- accuracy vs program length
- accuracy vs effective program length
- intermediate-state probe accuracy
- unique parameter count
- training time
- inference time

## Primary comparison

μCode Oracle vs Single Executor.

## Success criterion

A reproducible long-program generalization advantage over the
single-executor model across at least 3 seeds.

## Failure criterion

No meaningful advantage after matched training and adequate convergence.