# Grounded Prompting

ID: doc-07

## Definition

A grounded prompt tells the model to answer only from supplied evidence and to cite stable document IDs.
The evidence-first variant asks the model to list supporting facts before composing the final answer.

## Trade-offs

More instructions can improve format adherence, but they consume context and may conflict with one another.

## Example

The response contract contains an answer string, a citation list, and an abstained boolean.

## Limitations

Prompt instructions reduce unsupported answers but do not provide a hard factual guarantee.
