# Dataset v1.1.0: stratified split and review

`data/eval/rag_quality_v1.1.json` is `rag_quality_v1.json` (1.0.0, unchanged) plus a
dev/holdout split and a review label on every case. The holdout is frozen in
`data/eval/rag_quality_v1.1.holdout-lock.json`.

## How the split was made

```bash
python scripts/stratified_split.py \
  --source data/eval/rag_quality_v1.json \
  --output data/eval/rag_quality_v1.1.json \
  --version 1.1.0 --seed 20260928 \
  --reviewer "claude-opus-5-5 (AI review requested by repository owner)" \
  --reviewed-at 2026-09-28T12:00:00+08:00 \
  --reject rag-025 \
  --description "<see the file>"
rag-quality dataset freeze-holdout --dataset data/eval/rag_quality_v1.1.json
```

Re-running the script with the same arguments reproduces the file byte for byte.

- Strata: (first tag, difficulty). Each tag contributes one third of its cases to the
  holdout (single_document 8, multi_document 4, explicit_oos 2,
  plausible_unsupported 2), spread over difficulty by largest remainder.
- Cases whose review is not approved are never drawn into the holdout.
- Result: dev 32, holdout 16; holdout hash `fd58b56b07fec383a51e1899ec9edb7d0be17451acca6deabe319b9dc0d60571`.

| Stratum | Holdout | Case IDs |
|---|---|---|
| single_document / easy | 4 | rag-001, rag-007, rag-016, rag-021 |
| single_document / medium | 4 | rag-006, rag-010, rag-018, rag-024 |
| multi_document / hard | 4 | rag-026, rag-027, rag-031, rag-034 |
| explicit_oos / easy | 1 | rag-042 |
| explicit_oos / medium | 1 | rag-039 |
| plausible_unsupported / medium | 1 | rag-045 |
| plausible_unsupported / hard | 1 | rag-044 |

## Review

The review was performed by an AI model at the repository owner's request, not by an
independent human annotator. It checked:

1. Every `reference_evidence` sentence appears verbatim in one of the case's
   `expected_document_ids` (automated; all 36 answerable cases pass).
2. Tag and answerability agree (single_document has one expected document,
   multi_document at least two, unanswerable cases have none; all pass).
3. The reference answer is supported by the evidence and the question is unambiguous
   (manual reading of all 48 cases).
4. For the 12 unanswerable cases, the 1,040-word corpus was searched for numbers,
   thresholds, benchmarks, prices, measurements, and the out-of-scope topics; none
   contains an answer.

| Case | Status | Note |
|---|---|---|
| rag-025 | rejected | The reference answer says "RAG begins by retrieving external evidence", which contradicts doc-01's stage order (ingestion comes first); "reproducible" in the question is not supported by either evidence sentence. Kept in dev, excluded from the holdout. |
| rag-032 | approved | "Evaluation split" can be misread as a dataset split; the reference answer covers both separate metrics and the unanswerable subset, so scoring is unaffected. |
| rag-043 | approved | doc-03 mentions a 400-character chunk as an example but states no chunk size is optimal for every case, so "unanswerable" is correct; the example number is a deliberate distractor. |
| all others | approved | Evidence supports the reference answer; labels are consistent. |

## Limits

- **The holdout is not pristine.** All 48 cases, including the 16 now in the holdout,
  were used by experiments before the split (see `docs/artifacts/`). The holdout is
  held out from future tuning only; results on it are not evidence of performance on
  unseen questions.
- Sixteen cases give wide confidence intervals; report holdout numbers with their
  case counts and do not rank close configurations on them.
- Archived experiments and example configs still reference 1.0.0, whose content hash
  is unchanged; comparisons across 1.0.0 and 1.1.0 runs are refused by the paired
  comparison guard because the dataset versions differ.
