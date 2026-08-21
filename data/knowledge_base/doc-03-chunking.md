# Text Chunking

ID: doc-03

## Definition

Chunking divides a normalized document into retrieval units that fit the embedding and prompt limits.
Chunk overlap repeats boundary text so facts split near an edge can remain recoverable.

## Trade-offs

Small chunks improve targeting but lose context, while large chunks preserve context but may dilute similarity scores.

## Example

A 400-character chunk with 50-character overlap advances by 350 characters after each split.

## Limitations

One chunk size cannot be assumed optimal for every document shape, language, or question type.
