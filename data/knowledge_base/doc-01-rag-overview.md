# RAG System Overview

ID: doc-01

## Definition

Retrieval-augmented generation (RAG) retrieves external evidence before an LLM writes an answer.
A RAG request normally includes ingestion, retrieval, prompt construction, generation, and evaluation.

## Trade-offs

RAG can update knowledge without retraining the generator, but it adds retrieval latency and another source of failure.

## Example

A support assistant retrieves two product-policy passages and cites them in its response.

## Limitations

RAG does not guarantee correctness when the corpus is incomplete, retrieval misses evidence, or the generator ignores context.
