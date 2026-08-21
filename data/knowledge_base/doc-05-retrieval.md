# Dense Retrieval

ID: doc-05

## Definition

Dense retrieval ranks chunks by cosine similarity between query and chunk embeddings.
Recall@K measures whether at least one expected document appears in the first K retrieval results.

## Trade-offs

Increasing top K can improve recall, but it consumes prompt tokens and may introduce distracting evidence.

## Example

If an expected document is ranked third, Recall@3 is one and reciprocal rank is one third.

## Limitations

Dense retrieval can miss exact identifiers and rare terms that lexical retrieval handles well.
