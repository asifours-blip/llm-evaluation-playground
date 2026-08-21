# Embeddings and Caching

ID: doc-04

## Definition

An embedding maps text to a numeric vector so semantic similarity can be computed.
Embedding caches must include the model name, normalized text hash, and preprocessing version in the key.

## Trade-offs

Larger embedding models may improve semantic separation, but they increase latency, storage, and migration cost.

## Example

The vector for a cached chunk is reused only when both its text hash and embedding-model identifier match.

## Limitations

Cosine similarity reflects vector geometry, not factual correctness or permission to answer.
