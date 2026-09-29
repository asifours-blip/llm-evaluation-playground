# Complete task evaluation (offline)

`rag-quality task-eval` scores **recorded** complete-agent tasks. It does not execute an upstream agent or make a model request. The checked-in `data/task_eval/` suite contains exactly 12 **synthetic demo** tasks. Its scores demonstrate the evaluator, not customer-service or product-studio performance.

## Input contract

The trusted `suite.json` uses `schema_version: task-eval/v1`, `suite_id`, `suite_version`, `source_kind`, and `tasks`. Each task owns `task_id`, `scenario`, an evidence catalog (`evidence_id`, `source_version`, `role: authoritative|conflicting`), and `expected` fields: `artifact`, `citation_ids`, `final_state`. `suite.lock.json` holds the canonical SHA-256 of the complete suite. Updating expectations requires an explicit new suite version and lock; old results remain tied to their original hash.

The separate, untrusted `observations.json` uses `schema_version: task-observations/v1`, the suite ID/version/hash, `run_id`, `producer_name`, `producer_version`, and `tasks`. Each task observation may contain `retrieved_evidence_ids`, `artifact`, `citation_ids`, and `final_state`. An exporter cannot supply `expected` or `passed`: unknown fields are rejected. It must provide snapshots from the same task context; a claimed state alone cannot prove an external side effect. Strip personal/order identifiers or use test identifiers before export.

Input validation refuses malformed schema, duplicate or unknown task IDs, and mismatched suite identity/hash. Unknown or cross-task evidence IDs in retrieval/citations are **scored as observations**, not rejected: they expose an invalid citation or missing authoritative evidence. The suite is the only source of truth for which evidence is authoritative.

## Metrics and diagnosis

The primary rate is `successful_tasks / total_tasks` over **every** suite task, including absent or partial observations. Reports separately show received observations, complete-observation coverage, success among complete observations, and step pass rate. Retrieval, artifact, citations, and final state are checked independently. Citation checks require expected IDs, authoritative ownership, and presence in the retrieved set. Artifact/state assertions compare JSON values with strict types, so `false` never equals `0`.

Diagnostic tags describe **rule mismatches**, not proven upstream root causes. `retrieval_missing_evidence` means expected evidence ID was absent; `reasoning_or_selection_error` means expected evidence was present but artifact fields differed; `knowledge_conflict` marks an artifact mismatch when a catalogued conflicting source was cited. `missing_required_field`, `invalid_citation`, `state_mismatch`, `incomplete_observation`, and `undetermined` describe their corresponding observed conditions. A retrieval miss and an answer mismatch can co-occur; no tag alone proves why the upstream agent behaved that way.

## Reproduce the demo

```powershell
rag-quality task-eval validate --suite data/task_eval/suite.json --observations data/task_eval/observations-demo.json
rag-quality task-eval run --suite data/task_eval/suite.json --observations data/task_eval/observations-demo.json --output artifacts/task-demo
rag-quality task-eval compare --baseline artifacts/task-demo/<baseline>.json --candidate artifacts/task-demo/<candidate>.json
rag-quality serve --workspace .
```

Open the workbench's **Complete task evaluation** section and enter the generated report JSON path. Its result and comparison endpoints are read-only and use the existing workspace path restriction. Comparison requires identical frozen suite hash and task IDs; changing the suite invalidates a direct score delta. Real upstream integration is optional and requires its own reviewed, versioned exporter plus actual state evidence. This repository does not claim such integration from the synthetic fixture.
