# Validated Judge Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live RAG benchmark recover from semantically invalid structured answers and evaluate candidate answers using an explicit 1–5 reference-answer rubric before a fresh, budget-gated run and held-out human calibration.

**Architecture:** Keep generation, judging, and calibration separate. The provider retries only bounded semantic-format failures within the pre-existing request envelope; the scalar Judge prompt defines score caps for omitted core reference content. A new live configuration continues to use `deepseek-v4-flash`: verified peak-price preflight shows a complete `deepseek-v4-pro` 96-arm judge would exceed the ¥20 cap. A fresh human holdout is collected only after the rerun and before Judge scores are unlocked.

**Tech Stack:** Python 3.11, Pydantic, pytest, OpenAI-compatible provider, DeepSeek pricing configuration, SQLite WAL.

---

### Task 1: Define the strict scalar Judge rubric

**Files:**
- Modify: `src/rag_quality_lab/metrics/judge.py:65-80`
- Modify: `tests/metrics/test_judge.py:28-42`

- [ ] **Step 1: Write the failing rubric-contract test**

```python
def test_scalar_prompt_caps_scores_when_core_reference_content_is_missing() -> None:
    prompt = build_scalar_judge_prompt(
        question="How should a report diagnose failures?",
        reference_answer="Report P50 and distinguish retrieval misses.",
        candidate_answer="Store the commit SHA.",
        evidence=["E"],
    )

    assert "Score 1" in prompt
    assert "Score 2" in prompt
    assert "missing the central reference requirement" in prompt
    assert "must not receive a score of 4 or 5" in prompt
```

- [ ] **Step 2: Run the test and verify the expected red failure**

Run: `pytest tests/metrics/test_judge.py::test_scalar_prompt_caps_scores_when_core_reference_content_is_missing -v`

Expected: failure because the old prompt contains no explicit score caps.

- [ ] **Step 3: Implement the minimal fixed rubric**

```python
rubric = (
    "Score 5 only when the candidate answers the question and covers every core "
    "requirement in Reference. Score 4 only when it is mostly correct with a minor "
    "omission. Score 3 for a partially correct answer with a material omission. "
    "Score 2 when it is relevant but misses the central reference requirement. "
    "Score 1 when it is wrong, unsupported, or refuses an answerable question. "
    "A candidate missing the central reference requirement must not receive a score "
    "of 4 or 5."
)
```

- [ ] **Step 4: Run the focused and full metric tests**

Run: `pytest tests/metrics/test_judge.py -v`

Expected: all pass.

### Task 2: Make the existing bounded answer-repair contract reject null answers

**Files:**
- Modify: `src/rag_quality_lab/providers/openai_compatible.py:151-196`
- Modify: `tests/providers/test_openai_compatible.py:164-185`

- [ ] **Step 1: Write the failing provider test**

```python
def test_answer_contract_and_repair_forbid_null_answer(make_provider):
    session = FakeSession([
        answer_response('{"answer":null,"citations":[],"abstained":false}'),
        answer_response('{"answer":"fixed","citations":[],"abstained":false}'),
    ])
    response = make_provider(session=session).answer("q", [], model="m")
    assert response.parsed.answer == "fixed"
    assert "non-empty string" in session.calls[0]["json"]["messages"][0]["content"]
    assert "must not be null" in session.calls[1]["json"]["messages"][1]["content"]
```

- [ ] **Step 2: Run the test and verify the expected red failure**

Run: `pytest tests/providers/test_openai_compatible.py::test_answer_retries_semantically_invalid_repair_within_retry_cap -v`

Expected: failure because the current primary and repair contracts do not explicitly prohibit `answer: null`.

- [ ] **Step 3: Strengthen the existing primary/repair JSON contracts**

Require a non-empty string `answer` and explicitly forbid `null` in both the primary and repair prompts. Do not add a semantic retry loop: the existing two messages each already consume up to `max_retries + 1` network attempts, which is the preflight envelope of six generation requests. Do not change request token limits, retry counts, or error sanitization.

- [ ] **Step 4: Run focused provider tests**

Run: `pytest tests/providers/test_openai_compatible.py -v`

Expected: all pass, including existing one-repair behavior.

### Task 3: Preserve the live budget contract and prepare a corrected live run

**Files:**
- Create: `configs/live-deepseek-flash-strict-judge.example.yaml`
- Modify: `tests/experiments/test_runner.py:267-312`
- Modify: `docs/artifacts/pricing-verification-2026-08-21.json`

- [ ] **Step 1: Write the preflight regression test**

```python
def test_strict_judge_configuration_stays_under_live_budget() -> None:
    config = load_experiment_config("configs/live-deepseek-flash-strict-judge.example.yaml")
    decision = preflight_budget(
        planned=planned_calls(config, case_count=96),
        pricing=load_yaml_model(config.pricing_path, PricingConfig),
        budget=config.budget,
    )
    assert decision.allowed is True
    assert decision.buffered_cost < config.budget.hard_limit
```

- [ ] **Step 2: Run the test and verify the expected red failure**

Run: `pytest tests/experiments/test_runner.py::test_strict_judge_configuration_stays_under_live_budget -v`

Expected: failure because the corrected configuration does not yet exist.

- [ ] **Step 3: Add the corrected Flash configuration**

Use the verified `deepseek-2026-08-21.yaml`, `deepseek-v4-flash` for both chat and Judge, same 48-case dataset/2 retrieval configurations, `max_retries: 2`, `hard_limit: 20`, `preflight_fraction: 0.90`, and `safety_multiplier: 1.25`.

- [ ] **Step 4: Run configuration and preflight checks**

Run: `python -m rag_quality_lab.cli validate --config configs/live-deepseek-flash-strict-judge.example.yaml`

Run: `python -m rag_quality_lab.cli run --config configs/live-deepseek-flash-strict-judge.example.yaml --preflight-only`

Expected: validation succeeds; preflight reports `allowed: true` and a buffered cost below ¥18.

### Task 4: Verification and paid-run handoff

**Files:**
- Test: `tests/metrics/test_judge.py`
- Test: `tests/providers/test_openai_compatible.py`
- Test: `tests/experiments/test_runner.py`
- Create after explicit paid confirmation: `docs/artifacts/live-preflight-strict-judge-2026-08-21.json`

- [ ] **Step 1: Run static and regression checks**

Run: `ruff check src tests`

Run: `mypy --strict src`

Run: `pytest -q`

Expected: all checks pass.

- [ ] **Step 2: Write the no-cost preflight artifact**

Record verified pricing source/date, maximum HTTP attempts, unbuffered cost, 1.25× buffered cost, and an explicit `gate` value.

- [ ] **Step 3: Stop for a paid-run confirmation**

Do not call DeepSeek. Request the exact confirmation phrase `确认执行修复后真实 benchmark` only when the buffered preflight cost is below ¥18.

- [ ] **Step 4: After paid confirmation, run the corrected benchmark once**

Run the CLI with the `.env` key loaded into the child process only. Preserve the ¥20 hard cap and 1.5× preflight fuse. Stop immediately after the run and export a new 12-row, reference-answer human label template without Judge scores.

### Task 5: Fresh held-out calibration and reporting

**Files:**
- Create after independent labels: `docs/artifacts/calibrate-strict-judge-2026-08-21.json`
- Create only if all gates pass: `docs/artifacts/final-*.json`, `docs/artifacts/final-*.html`

- [ ] **Step 1: Validate the new label file**

Require exactly 12 rows, all scores/reasons present, immutable content hashes, manifest mapping equality, and a reviewer who did not see the new Judge scores.

- [ ] **Step 2: Apply the strict calibration gates**

Require within-one agreement ≥11/12, zero absolute score differences ≥2, and absolute mean signed difference ≤0.5. Report quadratic weighted κ without treating it as a separate veto.

- [ ] **Step 3: Stop on any failed gate**

Write `gate: fail`, reasons, and escalations. Do not generate a final report or commit a misleading evidence claim.

- [ ] **Step 4: Generate final evidence only on pass and zero case failures**

Generate the final JSON/HTML report only when the corrected run has no failed case-arms and the calibration gates pass; otherwise retain the pilot evidence and stop. Commit all audit artifacts locally only after final evidence exists. Do not push or open a PR.

## Self-review

- Scope coverage: Tasks 1–2 address the two observed root causes; Task 3 preserves the financial boundary; Tasks 4–5 enforce paid-run and calibration gates.
- No placeholders: every task names paths, commands, a success condition, and the relevant data contract.
- Type consistency: `JudgeVerdict` remains the existing `score/passed/reason` schema; no new persisted schema is needed for the corrected full rerun.
