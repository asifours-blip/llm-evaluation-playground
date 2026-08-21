# Candidate Reranking

ID: doc-06

## Definition

A reranker reorders a small candidate set using a query-document relevance model.
Reranking improves precision only when the first-stage retriever already includes useful evidence.

## Trade-offs

Cross-encoder rerankers inspect query-document pairs more deeply, but add one scoring operation per candidate.

## Example

A retriever returns 20 chunks, a reranker scores those 20, and the prompt receives the best 4.

## Limitations

A reranker cannot recover a relevant document that was absent from the candidate set.
