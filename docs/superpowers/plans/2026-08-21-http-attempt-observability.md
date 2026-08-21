# HTTP Attempt Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist and report the exact generation-and-Judge HTTP attempt count for every newly instrumented benchmark case, including retries, repairs, and isolated failures.

**Architecture:** A counter local to each OpenAI-compatible provider operation is passed into `_post_json` and incremented before every `session.post`. Typed provider responses and provider errors carry the operation count; the runner reduces those values to one optional case total, and reporting only emits an integer aggregate when every case is known.

**Tech Stack:** Python 3.11, Pydantic v2, requests-compatible session protocol, SQLite JSON payloads, Jinja2, pytest, Ruff, strict mypy.

---

## File map

- Modify `src/rag_quality_lab/domain/models.py`: add backward-compatible optional count fields to `ProviderResponse` and `CaseResult`.
- Modify `src/rag_quality_lab/providers/openai_compatible.py`: count physical attempts per public provider operation and attach counts to success/error paths.
- Modify `src/rag_quality_lab/providers/fake.py`: declare zero physical HTTP requests.
- Modify `src/rag_quality_lab/experiments/runner.py`: persist exact completed and failed case totals.
- Modify `src/rag_quality_lab/reporting/report.py`: aggregate only complete case counts.
- Modify `src/rag_quality_lab/reporting/templates/report.html.jinja2`: show exact or unknown HTTP request count.
- Modify focused provider, runner, store, and report tests.
- Modify `README.md`: document the new report field without changing historical artifact claims.

### Task 1: Provider-level physical attempt counting

**Files:**
- Modify: `tests/providers/test_openai_compatible.py`
- Modify: `tests/providers/test_fake_provider.py`
- Modify: `src/rag_quality_lab/domain/models.py`
- Modify: `src/rag_quality_lab/providers/openai_compatible.py`
- Modify: `src/rag_quality_lab/providers/fake.py`

- [x] **Step 1: Write failing provider tests**

Extend the malformed-answer test and add an exhausted-retry test:

```python
def test_retry_and_repair_count_every_http_attempt(make_provider):
    session = FakeSession([
        FakeResponse(500, {"error": "retry"}, text="retry"),
        answer_response("not json"),
        answer_response('{"answer":"fixed","citations":[],"abstained":false}'),
    ])
    provider = make_provider(session=session, sleeper=lambda _: None)

    response = provider.answer("q", [], model="m")

    assert response.http_request_count == 3


def test_exhausted_retries_expose_attempt_count(make_provider):
    session = FakeSession([
        FakeResponse(500, {"error": "first"}, text="first"),
        FakeResponse(500, {"error": "second"}, text="second"),
    ])
    provider = make_provider(session=session, sleeper=lambda _: None)

    with pytest.raises(ProviderError) as error:
        provider.answer("q", [], model="m")

    assert error.value.http_request_count == 2
```

Also assert `FakeChatProvider.answer(...).http_request_count == 0` and `FakeJudgeProvider.judge(...).http_request_count == 0`.

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/providers/test_openai_compatible.py tests/providers/test_fake_provider.py -q
```

Expected: failures showing that `ProviderResponse` and `ProviderError` do not expose `http_request_count`.

- [x] **Step 3: Add typed fields and local counting**

Add these model fields:

```python
class ProviderResponse(BaseModel, Generic[ResponseT]):
    parsed: ResponseT
    usage: TokenUsage
    model: str
    latency_ms: float = Field(default=0, ge=0)
    http_request_count: int | None = Field(default=None, ge=0)
    raw: dict[str, Any] | None = None


class CaseResult(BaseModel):
    # existing fields remain unchanged
    http_request_count: int | None = Field(default=None, ge=0)
```

Add a private counter and an observation context in `openai_compatible.py`:

```python
@dataclass
class _RequestCounter:
    count: int = 0


@contextmanager
def _observe_http_requests() -> Iterator[_RequestCounter]:
    counter = _RequestCounter()
    try:
        yield counter
    except ProviderError as error:
        error.http_request_count = counter.count
        raise
```

`ProviderError.__init__` accepts `http_request_count: int | None = None`; only instrumented OpenAI-compatible error paths fill it. Each `answer`, `judge`, `pairwise`, and `embed` operation opens an observation scope. `_chat_completion` and `_post_json` accept the counter, and `_post_json` executes `counter.count += 1` immediately before `session.post`. Successful typed responses set `http_request_count=counter.count`. Fake chat and Judge responses set `http_request_count=0`.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the same provider test command. Expected: all provider tests pass.

- [x] **Step 5: Commit provider instrumentation**

```powershell
git add -- tests/providers/test_openai_compatible.py tests/providers/test_fake_provider.py src/rag_quality_lab/domain/models.py src/rag_quality_lab/providers/openai_compatible.py src/rag_quality_lab/providers/fake.py
git commit -m "feat: count physical provider HTTP attempts"
```

### Task 2: Runner persistence for success and isolated failure

**Files:**
- Modify: `tests/experiments/test_runner.py`
- Modify: `src/rag_quality_lab/experiments/runner.py`

- [x] **Step 1: Write failing runner tests**

Create instrumented test providers that return generation count `2`, Judge count `3`, and a failing Judge `ProviderError(..., http_request_count=2)`. Assert:

```python
assert completed.http_request_count == 5
assert judge_failed.http_request_count == 3  # generation 1 + failed Judge 2
```

Also assert a retrieval failure records `0`, while an uninstrumented provider exception records `None`.

- [x] **Step 2: Run runner tests and verify RED**

```powershell
python -m pytest tests/experiments/test_runner.py -q
```

Expected: count assertions fail because `CaseResult.http_request_count` is not populated.

- [x] **Step 3: Implement one case-total reducer**

Add a private helper:

```python
def _known_http_request_total(*counts: int | None) -> int | None:
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)
```

Completed results sum generation and Judge counts, using Judge count `0` when Judge is not configured. Failed results use phase-aware inputs:

- retrieval: `0`;
- generation: instrumented `ProviderError.http_request_count`, otherwise `None`;
- metrics: completed generation response count;
- Judge: completed generation response count plus instrumented Judge error count, otherwise `None`.

- [x] **Step 4: Run runner tests and verify GREEN**

Run the same runner test command. Expected: all runner tests pass.

- [x] **Step 5: Commit runner persistence**

```powershell
git add -- tests/experiments/test_runner.py src/rag_quality_lab/experiments/runner.py
git commit -m "feat: persist per-case HTTP attempt totals"
```

### Task 3: SQLite round trip and truthful report aggregation

**Files:**
- Modify: `tests/experiments/test_store.py`
- Modify: `tests/reporting/test_report.py`
- Modify: `src/rag_quality_lab/reporting/report.py`
- Modify: `src/rag_quality_lab/reporting/templates/report.html.jinja2`

- [x] **Step 1: Write failing persistence and report tests**

Set `http_request_count=3` in the store fixture and assert it survives the SQLite JSON round trip. Set the report fixture count to `3` and assert:

```python
assert payload["system"]["http_request_count"] == 3
assert payload["system"]["http_request_count_complete"] is True
assert "HTTP requests" in html
```

Create a copied experiment whose case count is `None` and assert the JSON aggregate is `None` with completeness `False`.

- [x] **Step 2: Run tests and verify RED**

```powershell
python -m pytest tests/experiments/test_store.py tests/reporting/test_report.py -q
```

Expected: report system fields and HTML label are absent.

- [x] **Step 3: Implement complete-or-unknown aggregation**

In `_system_metrics`:

```python
request_counts = [result.http_request_count for result in results]
request_count_complete = all(count is not None for count in request_counts)

return {
    # existing metrics remain unchanged
    "http_request_count": (
        sum(count for count in request_counts if count is not None)
        if request_count_complete
        else None
    ),
    "http_request_count_complete": request_count_complete,
}
```

Update the return annotation to permit `bool | None`. Add an HTML system card that renders the integer when complete and `unknown` otherwise.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the same store/report test command. Expected: all tests pass.

- [x] **Step 5: Commit persistence and reporting**

```powershell
git add -- tests/experiments/test_store.py tests/reporting/test_report.py src/rag_quality_lab/reporting/report.py src/rag_quality_lab/reporting/templates/report.html.jinja2
git commit -m "feat: report exact HTTP attempt totals"
```

### Task 4: Documentation and full verification

**Files:**
- Modify: `README.md`
- Verify: `.github/workflows/ci.yml`

- [x] **Step 1: Document the field and legacy behavior**

Add one concise README note: new instrumented reports expose exact generation-and-Judge HTTP attempts; historical artifacts retain `null` because retries cannot be reconstructed.

- [x] **Step 2: Run all quality gates**

```powershell
python -m ruff check .
python -m mypy --strict src
python -m pytest -q
python -m pytest -m "not live" --cov=rag_quality_lab.domain --cov=rag_quality_lab.config --cov=rag_quality_lab.metrics --cov=rag_quality_lab.experiments.budget --cov=rag_quality_lab.experiments.compare --cov-report=term-missing --cov-fail-under=90
rag-quality regression --fixture tests/fixtures/offline_baseline.json
```

Expected: Ruff clean, strict mypy clean, all tests pass, coverage at least 90%, and regression `passed: true`.

- [x] **Step 3: Verify compatibility and secrets**

Load the historical final JSON through `ExperimentRecord`/report paths as applicable, verify its published hashes have not changed, verify `.env` remains ignored, and scan tracked files for the exact local API key without printing it.

- [x] **Step 4: Commit documentation**

```powershell
git add -- README.md
git commit -m "docs: explain HTTP attempt observability"
```

### Task 5: Post-review final-evidence hardening

**Files:**
- Modify: `tests/reporting/test_report.py`
- Modify: `src/rag_quality_lab/reporting/report.py`

- [ ] **Step 1: Write a failing final-gate test**

Create an otherwise eligible live experiment, set its case `http_request_count` to `None`, provide eligible 12-label calibration, and assert `generate_reports(..., badge="final")` raises a `ValueError` mentioning HTTP request evidence. Verify that the same experiment still generates a pilot report.

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m pytest tests/reporting/test_report.py::test_final_report_requires_complete_http_request_evidence -q
```

Expected: FAIL because current final validation does not reject the unknown count.

- [ ] **Step 3: Add the minimal final gate**

Extend `_validate_final_evidence`:

```python
if any(result.http_request_count is None for result in experiment.case_results):
    raise ValueError("final evidence requires complete HTTP request counts")
```

- [ ] **Step 4: Verify focused and full quality gates**

Run the report tests, Ruff, strict mypy, all tests, focused coverage, and offline regression commands from Task 4. Expected: every gate passes.

- [ ] **Step 5: Commit the review fix**

```powershell
git add -- tests/reporting/test_report.py src/rag_quality_lab/reporting/report.py docs/superpowers/specs/2026-08-21-http-attempt-observability-design.md docs/superpowers/plans/2026-08-21-http-attempt-observability.md
git commit -m "fix: require HTTP evidence for final reports"
```

## Plan self-review

- Every design requirement maps to a task and a focused test, including the post-review final gate.
- Field names are consistent: `http_request_count` and `http_request_count_complete`.
- Unknown and zero remain distinct through provider, runner, SQLite, report, and final-evidence validation layers.
- No paid call, matrix expansion, unrelated refactor, or historical artifact rewrite is included.
