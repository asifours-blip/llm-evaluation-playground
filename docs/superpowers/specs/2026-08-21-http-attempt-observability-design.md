# HTTP attempt observability design

## Purpose

Make the physical HTTP request-attempt count auditable for every new benchmark run. The count must include transport retries and structured-output repair calls, survive case isolation and SQLite persistence, and appear in canonical JSON/HTML reports. Existing reports that predate the field must remain readable and must continue to report the count as unknown rather than infer a value.

This change does not reinterpret the completed 96 case-arm run and does not authorize another paid run. A new exact count requires a new explicitly confirmed live benchmark after this instrumentation is merged.

## Root cause

`OpenAICompatibleProvider._post_json` owns the retry loop, but returns only the decoded response payload. `ProviderResponse` therefore carries token usage and latency but no transport-attempt metadata. `runner.py` persists selected fields from `ProviderResponse` into `CaseResult`, so the attempt count is already lost before SQLite and reporting see the result. Counting completed provider operations cannot recover retries or JSON repair calls.

## Considered approaches

### A. Per-operation counter propagated through typed results — selected

Create a request counter local to each `answer`, `judge`, and `pairwise` operation. `_post_json` increments it immediately before every `session.post`, so network failures, retryable status responses, successful responses, and repair calls are all counted. Successful operations expose the count through `ProviderResponse`; sanitized `ProviderError` exposes the count for failed operations. The runner computes one optional per-case total and the report aggregates only complete totals.

This remains correct under the runner's thread pool because no counter is shared between operations.

### B. Add request count to `TokenUsage` — rejected

This is mechanically smaller, but transport attempts are not token usage. Failed requests may have no provider token usage, while a repair call may contribute both tokens and another HTTP attempt. Combining the concepts would make cost and transport observability harder to reason about.

### C. Wrap the HTTP session with a process-wide counter — rejected

A global counter cannot reliably attribute attempts to concurrent case arms. Sampling before and after an operation would include requests made by other worker threads.

## Data model

`ProviderResponse` gains:

- `http_request_count: int | None = None`

`None` means that a provider implementation does not expose a trustworthy physical count. The deterministic fake providers explicitly set `0` because they perform no HTTP requests. The OpenAI-compatible provider always returns a non-negative exact value.

`ProviderError` gains:

- `http_request_count: int | None = None`

The OpenAI-compatible transport and parsing paths set this value from their local counter before raising a provider error. An error raised by an uninstrumented third-party provider remains unknown instead of being misreported as zero. This covers exhausted retries, authentication failures, invalid provider payloads, and a failed repair call without leaking the API key.

`CaseResult` gains:

- `http_request_count: int | None = None`

The runner stores the sum of generation and Judge counts for a completed case. For isolated failures it stores every count known up to the failure point. If a third-party provider raises an uninstrumented exception, the value remains `None`; the report must not under-count it.

## Data flow

1. `answer`, `judge`, or `pairwise` creates a local counter at operation entry.
2. Every `_post_json` attempt increments the counter before calling the session.
3. A structured-output repair reuses the same counter, so primary and repair attempts are combined.
4. Success returns the count in `ProviderResponse`.
5. A `ProviderError` is re-raised with the same count.
6. `runner.py` computes the case total for both completed and isolated-failure paths.
7. SQLite needs no schema migration because `CaseResult` is already stored as versioned JSON in `payload_json`.
8. Reporting emits:
   - `http_request_count`: exact integer when every case total is known, otherwise `null`.
   - `http_request_count_complete`: boolean explaining whether aggregation is complete.
9. Generating a new report with `badge=final` requires every case count to be known. A live run with incomplete transport evidence may still produce a pilot report.

## Backward compatibility

- Pydantic defaults both new persisted fields to `None`, so old SQLite rows and JSON artifacts still load.
- Existing final artifact files remain byte-for-byte unchanged and retain their explicit observability caveat. Regenerating a historical uninstrumented experiment under the new code may produce a pilot report, but cannot mint a new `final` badge without exact request evidence.
- Fake providers set `0`, preventing offline reports from becoming spuriously unknown.
- Custom providers that have not adopted the field remain supported but cannot produce an exact aggregate.

## Failure behavior

- Attempts are counted before the request so a connection exception still counts as a physical attempt.
- Authentication and non-retryable HTTP failures count the received request once.
- Exhausted retry loops report the full attempted count.
- Retrieval failures before provider invocation report zero.
- Generation failures with an uninstrumented exception report unknown.
- Metrics failures retain the already completed generation count.
- Judge failures retain the generation count and add the failed Judge operation's instrumented count.

## Test design

Tests are written before production changes and must first fail because the fields do not exist.

1. OpenAI-compatible answer: one retry plus one repair produces an exact count of three.
2. Exhausted retry error: the sanitized exception exposes all attempted requests.
3. Fake chat and Judge providers report zero HTTP requests.
4. Runner completed case: generation and Judge counts are summed and persisted.
5. Runner isolated Judge failure: successful generation plus failed Judge attempts are preserved.
6. Report aggregation: complete case counts produce an integer; any legacy/unknown case produces `null` and `http_request_count_complete=false`.
7. SQLite round trip preserves the optional per-case field.
8. Final gate: an otherwise eligible live experiment with an unknown case count is rejected, while pilot generation remains allowed.

After focused tests pass, run Ruff, strict mypy, the complete 111+ test suite, focused coverage at 90% or higher, and the deterministic regression gate.

## Non-goals

- No new paid benchmark is started by this change.
- No attempt is made to reconstruct the old run's physical count.
- Embedding HTTP calls are not added to the benchmark count because the supported live configuration uses the deterministic local embedding provider and the existing preflight scopes generation and Judge calls only.
- The original 384 case-arm aspiration versus the approved 96 case-arm budgeted run is recorded as a separate scope decision, not silently changed here.

## Self-review

- No placeholders or unresolved implementation choices remain.
- The selected design is concurrency-safe and distinguishes unknown from zero.
- Backward compatibility does not weaken truthfulness: legacy counts remain unknown.
- All changed lines will trace to the missing HTTP-attempt evidence requirement.
