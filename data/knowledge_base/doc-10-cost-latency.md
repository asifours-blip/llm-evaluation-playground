# Cost and Latency Controls

ID: doc-10

## Definition

A conservative preflight estimate multiplies bounded token counts by verified prices and a safety multiplier.
Latency reports should include mean, P50, and P95 rather than only an average.

## Trade-offs

Parallel requests reduce wall-clock time but increase rate-limit pressure and complicate ordered persistence.

## Example

A runner refuses to start when its buffered estimate exceeds ninety percent of the configured hard budget.

## Limitations

Token estimates and published prices can drift, so live runs must record actual usage and pricing verification dates.
