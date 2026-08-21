# RAG Evaluation

ID: doc-08

## Definition

Retrieval metrics and generation metrics must be reported separately to localize failures.
MRR rewards systems that place the first relevant result near the top of the ranking.

## Trade-offs

Reference metrics are repeatable and cheap, while model judges cover nuance but introduce cost and evaluator bias.

## Example

A run can have high Recall@5 and low answer correctness, showing that generation failed despite available evidence.

## Limitations

Aggregate scores hide case-level regressions unless reports preserve individual outcomes.
