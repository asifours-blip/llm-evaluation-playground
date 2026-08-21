# Reproducible Deployment

ID: doc-12

## Definition

A reproducible run records the commit SHA, dataset hash, prompt hash, configuration, model IDs, and random seed.
SQLite WAL mode lets readers continue while a coordinator serializes experiment writes.

## Trade-offs

Local SQLite keeps deployment simple, but it is not a substitute for a distributed high-write database.

## Example

The runner writes case results first, then generates static JSON and HTML reports from the committed experiment record.

## Limitations

Reproducibility metadata cannot compensate for a provider that silently changes a model behind the same identifier.
