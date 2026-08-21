# Failure Taxonomy

ID: doc-11

## Definition

A retrieval miss occurs when expected evidence is absent from the retrieved context.
A generation failure occurs when evidence is present but the answer is incorrect, unsupported, or malformed.

## Trade-offs

Fine-grained labels improve diagnosis, but annotation takes longer and ambiguous cases need adjudication.

## Example

When Recall@K is zero, the case is retrieval-limited; when Recall@K is one and correctness fails, it is generation-limited.

## Limitations

Some failures have multiple causes, so a taxonomy should preserve evidence instead of forcing certainty.
