# RAG Quality Lab Implementation Plan

> **Status note (2026-08-21):** This file preserves the original execution plan and its unchecked boxes are not the live project tracker. M1 is verified; M2 code paths are implemented, while a paid run and independent human labels remain external prerequisites. Current commands and evidence are in the root README.

> **Historical execution instruction:** Steps use checkbox (`- [ ]`) syntax from the original build plan.

**Goal:** Build a local-first, reproducible RAG evaluation platform with separated retrieval/generation metrics, persisted experiments, budget-safe live runs, baseline regression checks, and evidence-backed reports.

**Architecture:** Introduce a standard `src/rag_quality_lab` package while keeping the existing mock example compatible. Typed Pydantic domain objects cross module boundaries; provider calls and retrieval run concurrently, while a coordinator serializes SQLite writes in WAL mode. M1 delivers the offline closed loop, and M2 adds live-budget preflight, judge calibration, regression evidence, and resume documentation.

**Tech Stack:** Python 3.11+, Pydantic 2, requests, Jinja2, PyYAML, SQLite, pytest, pytest-cov, Ruff, mypy, GitHub Actions.

---

## Scope Map

This plan implements design-spec M1 and M2 only. FastAPI, Docker, a multi-version CI matrix, hosted vector databases, external evaluation frameworks, and batch inference remain M3 and are excluded.

### Target File Structure

```text
src/rag_quality_lab/
├── __init__.py
├── cli.py
├── config/
│   ├── __init__.py
│   └── loaders.py
├── domain/
│   ├── __init__.py
│   └── models.py
├── providers/
│   ├── __init__.py
│   ├── base.py
│   ├── fake.py
│   └── openai_compatible.py
├── retrieval/
│   ├── __init__.py
│   └── index.py
├── prompts/
│   ├── __init__.py
│   └── engine.py
├── metrics/
│   ├── __init__.py
│   ├── answer.py
│   ├── abstention.py
│   ├── calibration.py
│   ├── judge.py
│   └── retrieval.py
├── experiments/
│   ├── __init__.py
│   ├── budget.py
│   ├── compare.py
│   ├── runner.py
│   └── store.py
└── reporting/
    ├── __init__.py
    ├── report.py
    └── templates/report.html.jinja2
configs/
├── offline.yaml
├── live-deepseek.example.yaml
└── pricing/deepseek-2026-08-21.yaml
data/
├── knowledge_base/*.md
└── eval/rag_quality_v1.json
tests/
├── domain/
├── providers/
├── retrieval/
├── metrics/
├── experiments/
├── reporting/
└── test_cli.py
```

## Task 1: Establish the Package and Quality Tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/rag_quality_lab/__init__.py`
- Create: `tests/test_package.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing package test**

```python
from rag_quality_lab import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.2.0"
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_package.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'rag_quality_lab'`.

- [ ] **Step 3: Add package metadata and tool configuration**

`pyproject.toml` defines the `rag-quality-lab` project, Python `>=3.11`, runtime dependencies (`pydantic`, `requests`, `jinja2`, `PyYAML`, `python-dotenv`), dev dependencies (`pytest`, `pytest-cov`, `ruff`, `mypy`, `types-PyYAML`, `types-requests`), setuptools source layout, Ruff rules, mypy targets, and pytest markers `live` and `integration`.

`src/rag_quality_lab/__init__.py` contains:

```python
"""RAG Quality Lab: reproducible evaluation for LLM and RAG systems."""

__version__ = "0.2.0"
```

Add `.ragql/`, `artifacts/`, `*.sqlite3`, embedding caches, and live response caches to `.gitignore`.

- [ ] **Step 4: Install and verify GREEN**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/test_package.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- pyproject.toml .gitignore src/rag_quality_lab/__init__.py tests/test_package.py
git commit -m "build: establish rag quality lab package"
```

## Task 2: Define Dataset Domain Models and Loader

**Files:**
- Create: `src/rag_quality_lab/domain/__init__.py`
- Create: `src/rag_quality_lab/domain/models.py`
- Create: `src/rag_quality_lab/config/__init__.py`
- Create: `src/rag_quality_lab/config/loaders.py`
- Create: `tests/domain/test_dataset_models.py`
- Create: `tests/domain/test_dataset_loader.py`

- [ ] **Step 1: Write failing validation tests**

```python
import pytest
from pydantic import ValidationError

from rag_quality_lab.domain.models import Answerability, EvaluationCase, EvaluationDataset


def test_answerable_case_requires_documents_and_evidence() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="rag-001",
            question="What is chunk overlap?",
            reference_answer="It repeats boundary text.",
            answerability=Answerability.ANSWERABLE,
            expected_document_ids=[],
            reference_evidence=[],
            category="retrieval",
            difficulty="easy",
        )


def test_unanswerable_case_rejects_ground_truth_evidence() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="rag-002",
            question="Unsupported question",
            reference_answer="The corpus does not contain this information.",
            answerability=Answerability.UNANSWERABLE,
            expected_document_ids=["doc-01"],
            reference_evidence=["unsupported"],
            category="abstention",
            difficulty="medium",
        )


def test_dataset_rejects_duplicate_case_ids() -> None:
    case = EvaluationCase.answerable(
        id="rag-001",
        question="Q",
        reference_answer="A",
        expected_document_ids=["doc-01"],
        reference_evidence=["A"],
        category="retrieval",
        difficulty="easy",
    )
    with pytest.raises(ValidationError):
        EvaluationDataset(version="1.0.0", name="demo", cases=[case, case])
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/domain/test_dataset_models.py -q`

Expected: FAIL because the domain module does not exist.

- [ ] **Step 3: Implement the dataset models**

`models.py` defines:

```python
class Answerability(str, Enum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"


class EvaluationCase(BaseModel):
    id: str
    question: str
    reference_answer: str
    answerability: Answerability
    expected_document_ids: list[str] = Field(default_factory=list)
    reference_evidence: list[str] = Field(default_factory=list)
    category: str
    difficulty: Literal["easy", "medium", "hard"]
    tags: list[str] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    version: str
    name: str
    description: str = ""
    cases: list[EvaluationCase]
```

Use model validators for answerability invariants and unique IDs. Add `EvaluationCase.answerable` and `EvaluationCase.unanswerable` keyword-only constructors accepting the fields shown in the tests.

- [ ] **Step 4: Write and implement the JSON loader**

Test:

```python
def test_load_dataset_validates_json(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(valid_dataset_payload), encoding="utf-8")
    dataset = load_dataset(path)
    assert dataset.name == "demo"
    assert dataset.cases[0].id == "rag-001"
```

Implementation:

```python
def load_dataset(path: str | Path) -> EvaluationDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return EvaluationDataset.model_validate(payload)
```

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/domain/test_dataset_models.py tests/domain/test_dataset_loader.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- src/rag_quality_lab/domain src/rag_quality_lab/config tests/domain
git commit -m "feat: validate versioned evaluation datasets"
```

## Task 3: Define Experiment, Provider, Pricing, and Budget Configuration

**Files:**
- Modify: `src/rag_quality_lab/domain/models.py`
- Modify: `src/rag_quality_lab/config/loaders.py`
- Create: `tests/domain/test_experiment_config.py`

- [ ] **Step 1: Write failing configuration tests**

```python
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from rag_quality_lab.domain.models import BudgetConfig, ModelPrice, PricingConfig


def test_budget_requires_positive_hard_limit() -> None:
    with pytest.raises(ValidationError):
        BudgetConfig(currency="CNY", hard_limit=0)


def test_pricing_reports_staleness() -> None:
    pricing = PricingConfig(
        provider="deepseek",
        currency="CNY",
        verified_at=date.today() - timedelta(days=8),
        source_url="https://example.com/pricing",
        models={"flash": ModelPrice(input_cache_miss=1, output=2)},
    )
    assert pricing.is_stale(date.today(), max_age_days=7)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/domain/test_experiment_config.py -q`

Expected: FAIL because the configuration models do not exist.

- [ ] **Step 3: Implement typed configuration**

Add:

```python
class ProviderConfig(BaseModel):
    name: str
    base_url: AnyHttpUrl
    api_key_env: str
    chat_model: str
    embedding_model: str
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)


class RetrievalConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    top_k: int = Field(gt=0)
    prompt_variant: Literal["direct", "evidence_first"]


class ModelPrice(BaseModel):
    input_cache_hit: Decimal | None = Field(default=None, ge=0)
    input_cache_miss: Decimal = Field(ge=0)
    output: Decimal = Field(ge=0)


class PricingConfig(BaseModel):
    provider: str
    currency: str
    verified_at: date
    source_url: AnyHttpUrl
    models: dict[str, ModelPrice]


class BudgetConfig(BaseModel):
    currency: str = "CNY"
    hard_limit: Decimal = Field(gt=0)
    preflight_fraction: Decimal = Field(default=Decimal("0.90"), gt=0, le=1)
    safety_multiplier: Decimal = Field(default=Decimal("1.25"), ge=1)


class ExperimentConfig(BaseModel):
    name: str
    mode: Literal["mock", "live"]
    dataset_path: Path
    database_path: Path
    artifact_dir: Path
    random_seed: int = 42
    provider: ProviderConfig
    retrieval: list[RetrievalConfig]
    budget: BudgetConfig
    pricing_path: Path | None = None
```

Validate `chunk_overlap < chunk_size`, unique retrieval configurations, a pricing file for live mode, and matching budget/pricing currency.

- [ ] **Step 4: Add YAML loaders and verify GREEN**

```python
ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    return model_type.model_validate(payload)
```

Run: `python -m pytest tests/domain/test_experiment_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/domain/models.py src/rag_quality_lab/config/loaders.py tests/domain/test_experiment_config.py
git commit -m "feat: model reproducible experiment configuration"
```

## Task 4: Add the Controlled Knowledge Base and 48-Case Dataset

**Files:**
- Create: `data/knowledge_base/doc-01-rag-overview.md`
- Create: `data/knowledge_base/doc-02-ingestion.md`
- Create: `data/knowledge_base/doc-03-chunking.md`
- Create: `data/knowledge_base/doc-04-embeddings.md`
- Create: `data/knowledge_base/doc-05-retrieval.md`
- Create: `data/knowledge_base/doc-06-reranking.md`
- Create: `data/knowledge_base/doc-07-prompting.md`
- Create: `data/knowledge_base/doc-08-evaluation.md`
- Create: `data/knowledge_base/doc-09-abstention.md`
- Create: `data/knowledge_base/doc-10-cost-latency.md`
- Create: `data/knowledge_base/doc-11-failures.md`
- Create: `data/knowledge_base/doc-12-deployment.md`
- Create: `data/eval/rag_quality_v1.json`
- Create: `tests/domain/test_rag_quality_dataset.py`

- [ ] **Step 1: Write the failing corpus acceptance test**

```python
def test_rag_quality_dataset_has_required_distribution() -> None:
    dataset = load_dataset("data/eval/rag_quality_v1.json")
    assert len(dataset.cases) == 48
    assert sum(c.answerability == Answerability.ANSWERABLE for c in dataset.cases) == 36
    assert sum(c.answerability == Answerability.UNANSWERABLE for c in dataset.cases) == 12
    assert len(list(Path("data/knowledge_base").glob("*.md"))) == 12


def test_all_expected_documents_exist() -> None:
    dataset = load_dataset("data/eval/rag_quality_v1.json")
    document_ids = {path.name.split("-", 2)[0] + "-" + path.name.split("-", 2)[1]
                    for path in Path("data/knowledge_base").glob("*.md")}
    for case in dataset.cases:
        assert set(case.expected_document_ids) <= document_ids
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/domain/test_rag_quality_dataset.py -q`

Expected: FAIL because the corpus and dataset do not exist.

- [ ] **Step 3: Author the corpus and cases**

Each document has a stable `doc-NN` ID, title, concise definitions, trade-offs, one concrete example, and explicit limitations. Create 24 single-document answerable cases, 12 multi-document answerable cases, 6 explicit out-of-scope cases, and 6 plausible-but-unsupported cases. Every answerable case copies at least one exact supporting sentence into `reference_evidence`; unanswerable cases have no document IDs or evidence.

- [ ] **Step 4: Verify GREEN and inspect content**

Run: `python -m pytest tests/domain/test_rag_quality_dataset.py -q`

Run: `python -c "from rag_quality_lab.config.loaders import load_dataset; d=load_dataset('data/eval/rag_quality_v1.json'); print(d.name, len(d.cases))"`

Expected: tests PASS and command prints `rag-quality-v1 48`.

- [ ] **Step 5: Commit**

```bash
git add -- data/knowledge_base data/eval/rag_quality_v1.json tests/domain/test_rag_quality_dataset.py
git commit -m "data: add evidence-backed rag evaluation corpus"
```

## Task 5: Implement Provider Protocols and Deterministic Fake Providers

**Files:**
- Create: `src/rag_quality_lab/providers/__init__.py`
- Create: `src/rag_quality_lab/providers/base.py`
- Create: `src/rag_quality_lab/providers/fake.py`
- Modify: `src/rag_quality_lab/domain/models.py`
- Create: `tests/providers/test_fake_provider.py`

- [ ] **Step 1: Write failing provider contract tests**

```python
def test_fake_embedding_is_deterministic() -> None:
    provider = FakeEmbeddingProvider(dimensions=32)
    assert provider.embed(["chunk overlap"]) == provider.embed(["chunk overlap"])


def test_scripted_chat_returns_structured_answer() -> None:
    provider = FakeChatProvider(
        answers={"What is RAG?": StructuredAnswer(
            answer="Retrieval-augmented generation.",
            citations=["doc-01#chunk-000"],
            abstained=False,
        )}
    )
    response = provider.answer("What is RAG?", [], model="fake-model")
    assert response.parsed.answer.startswith("Retrieval")
    assert response.usage.total_tokens > 0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/providers/test_fake_provider.py -q`

Expected: FAIL because providers are missing.

- [ ] **Step 3: Implement provider protocols and response models**

Define `TokenUsage`, `StructuredAnswer`, `JudgeVerdict`, and generic `ProviderResponse[T]` Pydantic models. Define runtime-checkable `ChatProvider` and `EmbeddingProvider` protocols. Fake embeddings use deterministic token hashing into a fixed normalized vector. Fake chat uses an explicit question-to-answer map and raises a clear error for an unscripted question.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/providers/test_fake_provider.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/domain/models.py src/rag_quality_lab/providers tests/providers
git commit -m "feat: add deterministic provider interfaces"
```

## Task 6: Implement In-Memory Retrieval and Embedding Cache

**Files:**
- Create: `src/rag_quality_lab/retrieval/__init__.py`
- Create: `src/rag_quality_lab/retrieval/index.py`
- Create: `tests/retrieval/test_index.py`

- [ ] **Step 1: Write failing chunking and ranking tests**

```python
def test_chunk_ids_are_stable() -> None:
    chunks = chunk_document("doc-01", "abcdef", chunk_size=4, chunk_overlap=1)
    assert [chunk.id for chunk in chunks] == ["doc-01#chunk-000", "doc-01#chunk-001"]


def test_search_breaks_score_ties_by_chunk_id() -> None:
    provider = ConstantEmbeddingProvider([1.0, 0.0])
    index = InMemoryIndex.from_chunks(
        [Chunk(id="doc-02#chunk-000", document_id="doc-02", text="B"),
         Chunk(id="doc-01#chunk-000", document_id="doc-01", text="A")],
        provider,
    )
    hits = index.search("query", top_k=2)
    assert [hit.chunk.id for hit in hits] == ["doc-01#chunk-000", "doc-02#chunk-000"]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/retrieval/test_index.py -q`

Expected: FAIL because retrieval classes do not exist.

- [ ] **Step 3: Implement retrieval**

Define `Document`, `Chunk`, and `RetrievalHit` models. Implement deterministic character chunking, pure-Python cosine similarity, batch embedding, stable ordering, and a JSON embedding cache keyed by provider/model/content/chunking hashes. Reject `chunk_overlap >= chunk_size` and `top_k <= 0`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/retrieval/test_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/retrieval src/rag_quality_lab/domain/models.py tests/retrieval
git commit -m "feat: add deterministic in-memory retrieval"
```

## Task 7: Implement Retrieval, Answer, and Abstention Metrics

**Files:**
- Create: `src/rag_quality_lab/metrics/__init__.py`
- Create: `src/rag_quality_lab/metrics/retrieval.py`
- Create: `src/rag_quality_lab/metrics/answer.py`
- Create: `src/rag_quality_lab/metrics/abstention.py`
- Create: `tests/metrics/test_retrieval_metrics.py`
- Create: `tests/metrics/test_answer_metrics.py`
- Create: `tests/metrics/test_abstention_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
def test_recall_and_mrr_use_stable_document_ids() -> None:
    hits = [hit("doc-x", rank=1), hit("doc-b", rank=2), hit("doc-a", rank=3)]
    assert recall_at_k(hits, {"doc-a", "doc-b"}, k=2) == 0.5
    assert reciprocal_rank(hits, {"doc-a", "doc-b"}) == 0.5


def test_false_answer_and_over_abstention_rates() -> None:
    observations = [
        AbstentionObservation(expected_answerable=False, abstained=False),
        AbstentionObservation(expected_answerable=False, abstained=True),
        AbstentionObservation(expected_answerable=True, abstained=True),
        AbstentionObservation(expected_answerable=True, abstained=False),
    ]
    summary = summarize_abstention(observations)
    assert summary.false_answer_rate == 0.5
    assert summary.over_abstention_rate == 0.5
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/metrics -q`

Expected: FAIL because metrics do not exist.

- [ ] **Step 3: Implement metrics**

Implement document-level Recall@K, MRR, context-hit rate, normalized exact match, bilingual character/token F1, cosine semantic similarity, and a confusion-matrix-derived abstention summary with zero-denominator behavior defined as zero rather than an exception.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/metrics/test_retrieval_metrics.py tests/metrics/test_answer_metrics.py tests/metrics/test_abstention_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/metrics tests/metrics
git commit -m "feat: measure retrieval answers and abstention"
```

## Task 8: Implement Budget Preflight and Actual Cost Accounting

**Files:**
- Create: `src/rag_quality_lab/experiments/__init__.py`
- Create: `src/rag_quality_lab/experiments/budget.py`
- Create: `tests/experiments/test_budget.py`
- Create: `configs/pricing/deepseek-2026-08-21.yaml`

- [ ] **Step 1: Write failing budget tests**

```python
def test_preflight_uses_conservative_byte_token_estimate() -> None:
    estimate = estimate_tokens_upper_bound("中文")
    assert estimate == len("中文".encode("utf-8"))


def test_preflight_blocks_when_buffered_cost_exceeds_threshold() -> None:
    decision = preflight_budget(
        planned=[PlannedCall(model="pro", input_token_cap=3500, output_token_cap=256, count=192)],
        pricing=pricing(input_rate=3, output_rate=6),
        budget=BudgetConfig(hard_limit=Decimal("2.00")),
    )
    assert not decision.allowed
    assert decision.buffered_cost > Decimal("1.80")
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/experiments/test_budget.py -q`

Expected: FAIL because budget functions do not exist.

- [ ] **Step 3: Implement budget logic**

Use `Decimal` throughout. Implement pricing freshness, planned-call cost, 1.25 safety multiplication, the 90% preflight threshold, cache-hit/cache-miss actual-cost calculation, and a `BudgetLedger` that rejects scheduling when the next capped call could exceed the hard limit.

- [ ] **Step 4: Add verified pricing evidence and verify GREEN**

The pricing YAML records DeepSeek’s official source URL, `verified_at: 2026-08-21`, CNY currency, and peak-hour Flash/Pro rates as the conservative budget basis. It also records that off-peak rates are half of peak rates and the Beijing peak windows are 09:00–12:00 and 14:00–18:00.

Run: `python -m pytest tests/experiments/test_budget.py -q`

Expected: PASS, including the peak-rate RMB 15.575580 buffered full-plan example within one cent.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/experiments/budget.py src/rag_quality_lab/experiments/__init__.py tests/experiments/test_budget.py configs/pricing/deepseek-2026-08-21.yaml
git commit -m "feat: enforce live experiment budgets"
```

## Task 9: Implement the OpenAI-Compatible Provider

**Files:**
- Create: `src/rag_quality_lab/providers/openai_compatible.py`
- Create: `tests/providers/test_openai_compatible.py`

- [ ] **Step 1: Write failing retry and redaction tests**

```python
def test_authentication_failure_is_not_retried(provider, fake_session) -> None:
    fake_session.post.return_value = response(401, {"error": "bad key"})
    with pytest.raises(AuthenticationError):
        provider.answer("q", [], model="m")
    assert fake_session.post.call_count == 1


def test_rate_limit_honors_retry_after(provider, fake_session, sleeper) -> None:
    fake_session.post.side_effect = [
        response(429, {"error": "slow"}, headers={"Retry-After": "2"}),
        response(200, valid_answer_payload),
    ]
    provider.answer("q", [], model="m")
    sleeper.assert_called_once_with(2.0)


def test_error_never_contains_api_key(provider) -> None:
    with pytest.raises(ProviderError) as exc:
        provider._raise_sanitized("Bearer secret-key")
    assert "secret-key" not in str(exc.value)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/providers/test_openai_compatible.py -q`

Expected: FAIL because the provider does not exist.

- [ ] **Step 3: Implement HTTP behavior**

Use a `requests.Session`, environment-only API keys, `/chat/completions`, `/embeddings`, structured JSON validation, one JSON-repair attempt, retry classification, `Retry-After`, capped exponential backoff with jitter, response truncation, and token-usage normalization. Disable thinking mode by default using provider-compatible extra-body configuration without assuming every provider supports the same field.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/providers/test_openai_compatible.py -q`

Expected: PASS without a network call.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/providers/openai_compatible.py tests/providers/test_openai_compatible.py
git commit -m "feat: harden openai-compatible model access"
```

## Task 10: Implement Judge Validation and Human Calibration

**Files:**
- Create: `src/rag_quality_lab/metrics/judge.py`
- Create: `src/rag_quality_lab/metrics/calibration.py`
- Create: `tests/metrics/test_judge.py`
- Create: `tests/metrics/test_calibration.py`

- [ ] **Step 1: Write failing judge and calibration tests**

```python
def test_judge_verdict_enforces_pass_threshold() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(score=3, passed=True, reason="inconsistent")


def test_calibration_requires_twelve_labels() -> None:
    result = calibrate([HumanJudgePair(human_score=4, judge_score=4)] * 11)
    assert not result.blocking_eligible
    assert result.reason == "at least 12 labels are required"


def test_calibrated_judge_can_block() -> None:
    pairs = [HumanJudgePair(human_score=4, judge_score=4)] * 10 + [
        HumanJudgePair(human_score=3, judge_score=4),
        HumanJudgePair(human_score=5, judge_score=4),
    ]
    result = calibrate(pairs)
    assert result.within_one_rate == 1.0
    assert result.mean_absolute_error <= 1.0
    assert result.blocking_eligible
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/metrics/test_judge.py tests/metrics/test_calibration.py -q`

Expected: FAIL because judge/calibration modules are missing.

- [ ] **Step 3: Implement scalar and pairwise judging**

Implement rubric prompt builders, scalar JSON validation, A/B and B/A pairwise judging, `position_sensitive` exclusion, blind annotation export with model/config fields removed, human annotation import, exact agreement, within-one agreement, mean absolute error, and the eligibility thresholds from the design.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/metrics/test_judge.py tests/metrics/test_calibration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/metrics/judge.py src/rag_quality_lab/metrics/calibration.py src/rag_quality_lab/domain/models.py tests/metrics/test_judge.py tests/metrics/test_calibration.py
git commit -m "feat: calibrate structured llm judges"
```

## Task 11: Implement SQLite Experiment Persistence

**Files:**
- Create: `src/rag_quality_lab/experiments/store.py`
- Create: `tests/experiments/test_store.py`

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_store_enables_wal_and_busy_timeout(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    assert store.pragma("journal_mode").lower() == "wal"
    assert int(store.pragma("busy_timeout")) == 5000


def test_completed_experiment_round_trips(tmp_path: Path) -> None:
    store = ExperimentStore(tmp_path / "runs.sqlite3")
    experiment_id = store.create_experiment(example_identity())
    store.record_case_result(experiment_id, example_case_result())
    store.finish_experiment(experiment_id, ExperimentStatus.COMPLETED)
    loaded = store.get_experiment(experiment_id)
    assert loaded.status == ExperimentStatus.COMPLETED
    assert len(loaded.case_results) == 1
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/experiments/test_store.py -q`

Expected: FAIL because the store does not exist.

- [ ] **Step 3: Implement schema and repository**

Create tables `experiments`, `case_runs`, `retrieval_hits`, `metric_results`, `artifacts`, and `human_annotations`. Enable foreign keys, WAL, and a 5-second busy timeout. Store configuration and domain payloads as canonical JSON where normalized columns do not improve required queries. Enforce legal state transitions and expose read methods returning typed domain objects.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/experiments/test_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/experiments/store.py src/rag_quality_lab/domain/models.py tests/experiments/test_store.py
git commit -m "feat: persist reproducible experiments in sqlite"
```

## Task 12: Implement Prompt Rendering and Experiment Runner

**Files:**
- Create: `src/rag_quality_lab/prompts/__init__.py`
- Create: `src/rag_quality_lab/prompts/engine.py`
- Create: `src/rag_quality_lab/experiments/runner.py`
- Create: `tests/experiments/test_runner.py`
- Create: `configs/offline.yaml`

- [ ] **Step 1: Write failing end-to-end runner test**

```python
def test_offline_runner_persists_separate_retrieval_and_generation_metrics(tmp_path: Path) -> None:
    config = offline_config(tmp_path)
    result = run_experiment(config, fake_provider_bundle(), scripted_dataset())
    assert result.status == ExperimentStatus.COMPLETED
    assert result.case_results[0].metrics["retrieval_recall_at_k"] == 1.0
    assert "answer_f1" in result.case_results[0].metrics
    assert "false_answer_rate" in result.summary


def test_runner_marks_budget_exceeded_without_losing_completed_cases(tmp_path: Path) -> None:
    result = run_experiment(limited_live_config(tmp_path), capped_fake_bundle(), scripted_dataset())
    assert result.status == ExperimentStatus.BUDGET_EXCEEDED
    assert 0 < len(result.case_results) < len(scripted_dataset().cases)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/experiments/test_runner.py -q`

Expected: FAIL because prompts and runner do not exist.

- [ ] **Step 3: Implement prompt variants and hashing**

`direct` asks for a concise structured answer from supplied context. `evidence_first` asks the model to identify supporting citations before the answer. Both explicitly require abstention when context is insufficient. Canonical template text plus version produces a SHA-256 prompt hash.

- [ ] **Step 4: Implement the coordinator**

The runner loads documents and dataset, expands configurations, builds/caches indexes, submits provider work to a bounded `ThreadPoolExecutor`, computes metrics, and writes completed typed results through the coordinator thread. It records Git SHA, dirty state, dataset hash, prompt hashes, Python version, seed, configuration, usage, cost, failures, and artifact identities. Mock reports are labeled `mock` and cannot masquerade as live benchmark evidence.

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest tests/experiments/test_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- src/rag_quality_lab/prompts src/rag_quality_lab/experiments/runner.py configs/offline.yaml tests/experiments/test_runner.py
git commit -m "feat: run reproducible rag experiments"
```

## Task 13: Implement Baseline Comparison and Regression Gates

**Files:**
- Create: `src/rag_quality_lab/experiments/compare.py`
- Create: `tests/experiments/test_compare.py`
- Create: `configs/regression.yaml`

- [ ] **Step 1: Write failing comparison tests**

```python
def test_comparison_reports_metric_and_case_deltas() -> None:
    comparison = compare_experiments(baseline_experiment(), candidate_experiment())
    assert comparison.metric_deltas["retrieval_recall_at_k"].absolute == -0.1
    assert comparison.regressed_case_ids == ["rag-007"]


def test_uncalibrated_judge_metric_cannot_fail_gate() -> None:
    verdict = evaluate_regression(
        comparison_with_judge_drop(),
        rules=[RegressionRule(metric="judge_correctness", minimum_delta=0)],
        judge_calibration=uncalibrated_result(),
    )
    assert verdict.passed
    assert verdict.skipped_metrics == ["judge_correctness"]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/experiments/test_compare.py -q`

Expected: FAIL because comparison logic is missing.

- [ ] **Step 3: Implement comparison and gate rules**

Compare absolute and percentage metric deltas and case-level changes. Default offline rules gate Recall@K, MRR, false-answer rate, over-abstention rate, deterministic answer F1, and failure count. Judge rules are ignored unless calibration is blocking-eligible.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/experiments/test_compare.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/experiments/compare.py tests/experiments/test_compare.py configs/regression.yaml
git commit -m "feat: gate deterministic rag regressions"
```

## Task 14: Implement Static JSON and HTML Reports

**Files:**
- Create: `src/rag_quality_lab/reporting/__init__.py`
- Create: `src/rag_quality_lab/reporting/report.py`
- Create: `src/rag_quality_lab/reporting/templates/report.html.jinja2`
- Create: `tests/reporting/test_report.py`

- [ ] **Step 1: Write failing report tests**

```python
def test_report_exposes_identity_quality_cost_and_failures(tmp_path: Path) -> None:
    paths = generate_reports(example_experiment(), tmp_path)
    payload = json.loads(paths.json.read_text(encoding="utf-8"))
    html = paths.html.read_text(encoding="utf-8")
    assert payload["identity"]["dataset_hash"]
    assert payload["summary"]["false_answer_rate"] == 0.0
    assert "P95 latency" in html
    assert "mock" in html
    assert "Retrieved evidence" in html
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/reporting/test_report.py -q`

Expected: FAIL because reporting is missing.

- [ ] **Step 3: Implement reports**

Export canonical JSON and a self-contained Jinja2 HTML report with experiment identity, full configuration, metric summaries, category breakdowns, baseline deltas, cost/latency, failures, retrieval evidence, citations, judge reasons, calibration state, and an obvious `mock`, `pilot`, or `final` badge.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/reporting/test_report.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/reporting tests/reporting
git commit -m "feat: export evidence-rich evaluation reports"
```

## Task 15: Implement CLI and Legacy Mock Compatibility

**Files:**
- Create: `src/rag_quality_lab/cli.py`
- Create: `tests/test_cli.py`
- Modify: `examples/run_eval.py`
- Create: `configs/live-deepseek.example.yaml`

- [ ] **Step 1: Write failing CLI tests**

```python
def test_offline_cli_generates_database_and_reports(tmp_path: Path) -> None:
    result = run_cli(["run", "--config", str(offline_config_file(tmp_path))])
    assert result.exit_code == 0
    assert (tmp_path / "runs.sqlite3").exists()
    assert (tmp_path / "report.html").exists()


def test_live_cli_requires_explicit_confirmation(tmp_path: Path) -> None:
    result = run_cli(["run", "--config", str(live_config_file(tmp_path))])
    assert result.exit_code == 2
    assert "--confirm-live-run" in result.stderr
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: FAIL because CLI does not exist.

- [ ] **Step 3: Implement commands**

Use `argparse` with commands:

```text
rag-quality run --config FILE [--confirm-live-run]
rag-quality compare --database FILE --baseline ID --candidate ID
rag-quality report --database FILE --experiment ID --output DIR
rag-quality annotate export --database FILE --experiment ID_OR_LATEST_LIVE --count N --output FILE
rag-quality annotate import --database FILE --experiment ID_OR_LATEST_LIVE --input FILE
rag-quality calibrate --database FILE --experiment ID_OR_LATEST_LIVE
rag-quality regression --database FILE --baseline ID --candidate ID --rules FILE
rag-quality regression --fixture FILE
```

`examples/run_eval.py --mock` delegates to the offline config and prints the experiment/report paths. It does not retain duplicate evaluator logic.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_cli.py -q`

Run: `python examples/run_eval.py --mock`

Expected: tests PASS; compatibility command creates a labeled mock report without an API key.

- [ ] **Step 5: Commit**

```bash
git add -- src/rag_quality_lab/cli.py tests/test_cli.py examples/run_eval.py configs/live-deepseek.example.yaml
git commit -m "feat: expose evaluation workflows through cli"
```

## Task 16: Add Focused CI and Remove Superseded Implementation

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify or remove: legacy tests under `tests/test_*.py`
- Remove after replacement: `src/client`, `src/evaluator`, `src/prompt`, `src/rag`, `src/reporter`
- Modify: `examples/rag_demo.py`

- [ ] **Step 1: Run all old and new tests before removal**

Run: `python -m pytest -q`

Expected: PASS before legacy cleanup.

- [ ] **Step 2: Update remaining tests and examples to new package imports**

Preserve behavior that remains part of the design and delete tests tied only to superseded internal APIs. `examples/rag_demo.py` becomes a documented CLI example or is removed if it duplicates `rag-quality run`.

- [ ] **Step 3: Remove superseded code only after replacement is green**

Delete legacy packages listed above. Confirm no import references remain:

Run: `rg -n "from src\.|import src\." .`

Expected: no matches outside historical design documentation.

- [ ] **Step 4: Add CI and focused coverage gates**

GitHub Actions uses Python 3.11 and runs:

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy src/rag_quality_lab/domain src/rag_quality_lab/config src/rag_quality_lab/metrics src/rag_quality_lab/experiments/budget.py src/rag_quality_lab/experiments/compare.py
pytest -m "not live" --cov=rag_quality_lab.domain --cov=rag_quality_lab.config --cov=rag_quality_lab.metrics --cov=rag_quality_lab.experiments.budget --cov=rag_quality_lab.experiments.compare --cov-report=term-missing --cov-fail-under=90
rag-quality regression --fixture tests/fixtures/offline_baseline.json
```

Coverage enforcement is configured per pure-logic package rather than as a misleading repository-wide percentage.

- [ ] **Step 5: Verify the complete offline gate**

Run every CI command locally.

Expected: all commands pass with no warnings that indicate broken behavior.

- [ ] **Step 6: Commit**

Stage only the verified changed/deleted paths, then commit:

```bash
git commit -m "ci: enforce deterministic rag quality gates"
```

## Task 17: Produce Offline Evidence and Portfolio Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/interview_qa.md`
- Create: `docs/architecture.md`
- Create: `docs/resume_bullets.md`
- Create: `docs/artifacts/offline-summary.json`
- Create: `docs/artifacts/offline-report.html`
- Create: `tests/test_documented_commands.py`

- [ ] **Step 1: Generate a real offline artifact**

Run: `rag-quality run --config configs/offline.yaml`

Expected: a completed mock experiment, SQLite record, JSON report, and HTML report.

- [ ] **Step 2: Add a failing documentation command test**

```python
def test_readme_offline_command_succeeds(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rag_quality_lab.cli", "run", "--config", "configs/offline.yaml", "--artifact-dir", str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 3: Write evidence-based documentation**

README explains the problem, architecture, reproducibility identity, separated failure modes, exact offline/live commands, budget safeguards, metrics, limitations, and artifact provenance. Architecture documentation records module boundaries and data flow. Interview Q&A includes the no-vector-DB decision, judge calibration, abstention metrics, SQLite WAL, and why mock coverage is insufficient. Resume bullets clearly mark offline software evidence versus live model-quality evidence.

- [ ] **Step 4: Verify docs and commit**

Run: `python -m pytest tests/test_documented_commands.py -q`

Run: `git diff --check`

Expected: PASS and no whitespace errors.

```bash
git add -- README.md docs/architecture.md docs/interview_qa.md docs/resume_bullets.md docs/artifacts tests/test_documented_commands.py
git commit -m "docs: present reproducible rag evaluation evidence"
```

## Task 18: Run the M2 Live Benchmark and Human Calibration

**Files:**
- Create from commands: `docs/artifacts/live-final-summary.json`
- Create from commands: `docs/artifacts/live-final-report.html`
- Create from human action: `docs/artifacts/human-annotations.jsonl`
- Modify after evidence: `README.md`
- Modify after evidence: `docs/resume_bullets.md`
- Modify after evidence: `docs/interview_qa.md`

- [ ] **Step 1: Verify provider credentials without printing secrets**

Check only whether the configured environment variable exists. Never print its value.

- [ ] **Step 2: Re-verify official pricing**

Update a new dated pricing YAML only from the provider’s official price page. Record source URL and verification date. Do not edit historical price evidence.

- [ ] **Step 3: Run preflight without network calls**

Run: `rag-quality run --config configs/live-deepseek.example.yaml --preflight-only`

Expected: planned call count, token caps, unbuffered cost, buffered cost, hard limit, and `allowed: true`.

- [ ] **Step 4: Run the staged live experiment**

Run with explicit confirmation only after preflight passes:

```bash
rag-quality run --config configs/live-deepseek.example.yaml --confirm-live-run
```

Expected: Stages A–D complete or stop safely with persisted `budget_exceeded` status; actual cost remains within the hard limit.

- [ ] **Step 5: Export and complete blind human annotations**

Run:

```bash
rag-quality annotate export --database .ragql/experiments.sqlite3 --experiment latest-live --count 12 --output docs/artifacts/human-annotations.jsonl
```

A human fills all 12 scores without seeing model/config/judge fields. Import and calibrate:

```bash
rag-quality annotate import --database .ragql/experiments.sqlite3 --experiment latest-live --input docs/artifacts/human-annotations.jsonl
rag-quality calibrate --database .ragql/experiments.sqlite3 --experiment latest-live
```

- [ ] **Step 6: Publish only evidence-backed numbers**

Regenerate reports and update README/resume bullets with actual dataset size, model/config count, retrieval/generation/abstention results, judge agreement, P95 latency, and cost. If calibration fails, label judge metrics diagnostic and do not claim judge reliability.

- [ ] **Step 7: Run completion audit and commit**

Run all offline quality gates, validate artifact hashes, confirm no secrets or raw sensitive responses are staged, and inspect staged paths before committing:

```bash
git commit -m "docs: add validated live benchmark evidence"
```

## Final Verification

Run:

```bash
ruff check .
mypy src/rag_quality_lab/domain src/rag_quality_lab/config src/rag_quality_lab/metrics src/rag_quality_lab/experiments/budget.py src/rag_quality_lab/experiments/compare.py
pytest -m "not live" -q
python examples/run_eval.py --mock
rag-quality regression --fixture tests/fixtures/offline_baseline.json
git diff --check
git status --short --branch
```

M1 is complete only when every offline command passes and the generated artifacts prove the closed loop. M2 is complete only after the live benchmark and actual human annotation evidence exist; missing credentials or missing human labels must be reported as remaining external prerequisites rather than fabricated.
