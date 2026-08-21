# Abstention Evaluation

ID: doc-09

## Definition

Abstention means explicitly declining to answer when the controlled corpus lacks sufficient evidence.
False-answer rate is the fraction of unanswerable cases that receive a substantive answer.

## Trade-offs

Aggressive abstention reduces hallucinations but can reject answerable questions, so over-abstention must also be measured.

## Example

For an unsupported pricing question, the system returns an empty citation list and sets abstained to true.

## Limitations

String matching alone cannot reliably distinguish a cautious answer from a real abstention.
