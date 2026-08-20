# RAG Quality Lab Design Specification

**Status:** Approved for implementation planning

**Date:** 2026-08-21

**Repository:** `asifours-blip/llm-evaluation-playground`

**Target branch:** `codex/resume-ready-eval-platform`

## 1. Purpose

Upgrade the current LLM evaluation MVP into a resume-ready, local-first evaluation platform for LLM and RAG applications. The finished project must demonstrate a reproducible engineering workflow rather than a collection of disconnected demos:

1. version datasets and experiment configurations;
2. evaluate retrieval and generation separately;
3. record quality, abstention, latency, token usage, cost, and failures;
4. compare experiments against a baseline;
5. block deterministic regressions in CI;
6. support a budget-capped real-model benchmark;
7. back every public resume claim with committed or reproducible experiment evidence.

The project name used in documentation is **RAG Quality Lab**. The GitHub repository name does not need to change.

## 2. Current State and Problem

The existing repository already has an OpenAI-compatible client, three prompt strategies, a concurrent evaluator, semantic similarity, a simple numeric LLM judge, an HTML report, an experimental LangChain/Chroma RAG module, and 38 mock-heavy tests.

It is not yet strong enough as an interview project because:

- the 20 generic questions are a toy dataset without evidence annotations;
- retrieval and generation failures cannot be distinguished;
- the RAG demo is disconnected from the evaluator;
- experiments overwrite one HTML file and do not record configuration identity;
- the judge extracts the first digit from free-form text and is not calibrated against human labels;
- cost, tail latency, abstention, prompt hashes, dataset hashes, and Git commit identity are not recorded;
- no baseline comparison or CI regression gate exists;
- there are no GitHub Actions checks;
- the reported 97% coverage mainly proves mocked control flow, not evaluation validity.

## 3. Scope Tiers

### 3.1 M1: Closed-Loop MVP — Required

M1 is the minimum definition of “not a half-finished project.” It must include:

- a typed and validated dataset model;
- a typed experiment configuration;
- 12 controlled AI-engineering knowledge documents;
- 48 evaluation cases: 36 answerable and 12 unanswerable;
- an in-memory brute-force cosine retriever with cached embeddings;
- experiments over prompt, chunk size, and `top_k`; `chunk_overlap` remains explicit but fixed during the required sweep;
- retrieval, generation, abstention, latency, token, cost, and failure metrics;
- a SQLite experiment store with WAL mode;
- baseline selection and experiment comparison;
- a command-line entry point and compatibility for `python examples/run_eval.py --mock`;
- deterministic fake providers for a complete zero-cost offline run;
- static HTML and JSON reports;
- a single-version GitHub Actions workflow;
- focused tests and accurate setup documentation.

### 3.2 M2: Credibility Layer — Required Before Resume Publication

M2 must include:

- a real-model benchmark run with a preflight cost estimate and hard budget cap;
- 12 stratified, blind human labels for judge calibration;
- judge/human agreement reporting;
- an answer-order bias check for pairwise judging;
- deterministic regression thresholds and a CI quality gate;
- real experiment artifacts, screenshots, and conclusions;
- updated interview Q&A;
- Chinese and English resume bullets whose numbers are traceable to artifacts.

The project goal is complete only when M1 and M2 are both verified.

### 3.3 M3: Presentation Enhancements — Optional

M3 may be implemented only after M1 and M2 pass:

- FastAPI dashboard;
- Docker image;
- Python 3.10–3.12 CI matrix;
- fully bilingual documentation;
- adapters for external evaluation frameworks;
- provider batch-inference integration.

No M3 item may delay or redefine M1/M2 completion.

## 4. Non-Goals

The current project will not implement:

- authentication, authorization, accounts, or multi-tenancy;
- distributed task queues or multiple worker processes;
- production deployment infrastructure;
- a hosted vector database;
- agent/tool trajectory evaluation;
- document upload or knowledge-base management UI;
- automatic scraping of provider pricing pages;
- live paid-model calls in the default CI workflow.

## 5. Target Package and Module Boundaries

Production code will move from using `src` as an importable package to a standard source layout under `src/rag_quality_lab/`. Existing examples will be updated, and `examples/run_eval.py --mock` will remain a supported compatibility entry point.

The target module boundaries are:

| Module | Responsibility |
|---|---|
| `domain` | Pydantic domain models and invariants for datasets, cases, experiments, runs, metrics, and model responses |
| `config` | Load and validate dataset, experiment, provider, pricing, budget, and regression configuration |
| `providers` | Provider protocols plus OpenAI-compatible and deterministic fake implementations |
| `retrieval` | Chunking, embedding cache, in-memory index, cosine ranking, and retrieval traces |
| `prompts` | Versioned prompt templates and prompt hashing |
| `metrics` | Retrieval, generation, abstention, system, judge, and calibration metrics |
| `experiments` | Matrix expansion, run orchestration, budget preflight, SQLite persistence, baselines, and comparison |
| `reporting` | Static HTML and JSON export from stored experiment records |
| `cli` | User-facing offline run, live run, compare, report, annotate, and calibrate commands |

Units communicate through typed domain objects. Provider SDK responses, SQLite rows, and JSON files must be converted at module boundaries rather than passed through the system as unvalidated dictionaries.

## 6. Dataset Design

### 6.1 Knowledge Base

M1 will contain 12 concise Markdown documents under `data/knowledge_base/`. They cover controlled AI-engineering topics such as RAG ingestion, chunking, embeddings, retrieval, reranking, prompt design, evaluation, latency, token accounting, failure handling, and deployment trade-offs.

The documents are authored for this repository, so ground truth and citations remain stable. The benchmark must not depend on web content that can change after an experiment.

### 6.2 Evaluation Cases

The versioned dataset contains exactly 48 cases:

- 24 single-document answerable cases;
- 12 multi-document answerable cases;
- 6 explicit out-of-scope unanswerable cases;
- 6 plausible-but-unsupported unanswerable cases.

Each case includes:

```json
{
  "id": "rag-001",
  "question": "...",
  "reference_answer": "...",
  "answerability": "answerable",
  "expected_evidence_ids": ["doc-03#chunk-02"],
  "category": "retrieval",
  "difficulty": "medium",
  "tags": ["single-hop"]
}
```

Validation rejects duplicate IDs, missing evidence for answerable cases, evidence on unanswerable cases, empty questions or reference answers, unknown categories, and references to documents/chunks that do not exist.

The original 20-question dataset remains as a legacy generic-QA example but is not used to support the new resume claims.

## 7. Retrieval Design

M1 deliberately uses no external vector database.

1. Markdown documents are split with deterministic character-based chunking.
2. The required Stage A sweep fixes `chunk_overlap` at 50 characters so it does not create a fourth experimental axis.
3. Every chunk receives a stable ID derived from document ID and chunk position.
4. Embeddings are generated in batches through the configured embedding provider.
5. The cache key includes provider, model, document content hash, and chunking configuration.
6. Cached vectors are stored locally and excluded from Git.
7. Queries are embedded once per retrieval configuration.
8. Cosine similarity is computed in memory across all chunks.
9. Results are sorted deterministically by score and then chunk ID.
10. The top-K trace records rank, score, document ID, chunk ID, and text.

The expected scale is roughly one hundred chunks, where brute-force retrieval is fast, explainable, and easy to verify. Vector database support belongs to M3 or a later project.

## 8. Generation Output Contract

RAG answers use a structured output contract:

```json
{
  "answer": "...",
  "citations": ["doc-03#chunk-02"],
  "abstained": false
}
```

Providers that support native JSON output may enable it. All providers still pass through the same Pydantic validation. One repair attempt is allowed for malformed JSON; a second failure becomes a recorded case failure rather than an invented answer.

Unanswerable prompts explicitly require abstention. The evaluator never infers successful abstention solely from an empty answer; the validated `abstained` field and answer text are both checked.

## 9. Metric Design

### 9.1 Retrieval Metrics

- **Recall@K:** retrieved relevant evidence IDs divided by all expected evidence IDs.
- **MRR:** reciprocal rank of the first relevant evidence item, or zero if none is retrieved.
- **Context Hit Rate:** fraction of answerable cases with at least one expected evidence item in Top-K.

### 9.2 Generation Metrics

- normalized exact match for deterministic short answers;
- token/character F1 against the reference answer;
- embedding semantic similarity;
- LLM-judge correctness when a calibrated judge is enabled;
- LLM-judge faithfulness against retrieved context when a calibrated judge is enabled.

### 9.3 Abstention Metrics

- abstention accuracy;
- abstention precision, recall, and F1;
- false-answer rate: unanswerable cases that receive a non-abstaining answer;
- over-abstention rate: answerable cases that are incorrectly refused.

False-answer rate is a mandatory report field and a regression-gate input.

### 9.4 System Metrics

- success and failure counts;
- retry count;
- prompt, completion, cache-hit, cache-miss, and total tokens when supplied by the provider;
- estimated and actual cost;
- mean, P50, and P95 latency.

## 10. Judge Design and Human Calibration

The scalar judge returns validated JSON:

```json
{
  "score": 4,
  "passed": true,
  "reason": "The answer is correct but omits an important constraint."
}
```

`passed` must be `true` for scores 4–5 and `false` for scores 1–3. Scores outside 1–5, inconsistent `score`/`passed` combinations, empty reasons, and malformed payloads are rejected. A single repair attempt is permitted.

For M2, the CLI creates a blind annotation file containing 12 stratified outputs. Model name, prompt strategy, retrieval configuration, and automatic scores are hidden. A human assigns the same 1–5 rubric used by the judge.

Calibration reports:

- exact agreement rate;
- agreement within one point;
- mean absolute error.

The judge may become a blocking regression metric only when:

- there are at least 12 valid human labels;
- within-one agreement is at least 80%;
- mean absolute error is at most 1.0.

If calibration fails, judge scores remain diagnostic and cannot fail CI.

For final configuration comparison, the pairwise judge sees outputs A/B and B/A in separate calls. A preference counts only when both orders agree; otherwise the result is recorded as position-sensitive and excluded from the win-rate numerator.

## 11. Experiment Strategy

The system avoids a paid full Cartesian sweep.

### Stage A: Retrieval-Only Screening

Evaluate eight configurations:

```text
2 chunk sizes × 2 top_k values × 2 context prompt variants
```

No generation or judge calls occur. Rank configurations using retrieval metrics and retain the top three.

### Stage B: Low-Cost Pilot

Run the top three configurations on 12 stratified cases with one low-cost generation model:

```text
3 configurations × 12 cases = 36 generation calls
```

Use deterministic generation and abstention metrics to retain the top two configurations.

### Stage C: Final Benchmark

Run two configurations on all 48 cases with two models:

```text
2 configurations × 48 cases × 2 models = 192 generation calls
```

### Stage D: Pairwise Judge

Compare the two configurations within each model and reverse answer order:

```text
48 cases × 2 models × 2 answer orders = 192 judge calls
```

The planned maximum is 420 chat calls before retry allowance.

## 12. Pricing and Budget Safety

Pricing is explicit experiment input, not a permanent constant in source code. A pricing file includes provider, currency, verification timestamp, source URL, model rates, and cache-hit/cache-miss distinctions.

Rules:

- a live run refuses pricing data older than seven days;
- input tokens use a conservative UTF-8 byte-count upper bound during preflight;
- generation input is capped at 2,500 estimated tokens and output at 512 tokens;
- judge input is capped at 3,500 estimated tokens and output at 256 tokens;
- thinking/reasoning mode is disabled unless an experiment explicitly evaluates it;
- the preflight estimate is multiplied by 1.25 for retries and estimation error;
- no request is sent when the buffered estimate exceeds 90% of the configured budget;
- actual cost is accumulated from provider usage after each response;
- scheduling stops before the hard budget is exceeded;
- completed partial results are persisted with `budget_exceeded` status.

Using the official DeepSeek direct rates verified on 2026-08-21—Flash cache-miss input/output at RMB 1/2 per million tokens and Pro at RMB 3/6—the current 420-call plan has a conservative buffered estimate of approximately RMB 4.74, excluding embeddings. This value is evidence for the design only; every live run must re-verify current official pricing.

SiliconFlow may be used for a free development model and free BGE embeddings when the account is eligible. Its fixed free-model rate limits are treated as an operational constraint, not a cost guarantee for other hosted models.

## 13. Experiment Identity and Persistence

Every experiment records:

- experiment ID, name, timestamps, and lifecycle status;
- Git commit SHA and dirty-worktree flag;
- dataset content hash;
- prompt template hashes;
- complete provider, model, retrieval, generation, metric, budget, and price configuration;
- Python and dependency versions;
- random seed;
- per-case retrieval trace, generated answer, metrics, errors, usage, latency, and cost;
- artifact paths and hashes.

SQLite tables are:

- `experiments`;
- `case_runs`;
- `retrieval_hits`;
- `metric_results`;
- `artifacts`;
- `human_annotations`.

SQLite uses `PRAGMA journal_mode=WAL` and `PRAGMA busy_timeout=5000`. Worker threads perform provider calls and metric computation. The coordinator serializes writes on the main thread. Reports and the optional M3 dashboard use read-only connections.

Experiment states are `pending`, `running`, `completed`, `failed`, and `budget_exceeded`. An interrupted experiment remains inspectable and is never presented as completed.

## 14. Baselines and Regression Gates

The CLI can mark one completed experiment as the baseline for a named dataset/configuration family. Comparisons report absolute and percentage deltas plus the cases responsible for changes.

Default CI remains offline and deterministic. It runs a committed fake-provider fixture and gates:

- retrieval Recall@K;
- MRR;
- false-answer rate;
- over-abstention rate;
- deterministic answer F1;
- failure count.

Live-model scores are reported but do not run in default CI. Calibrated judge metrics may be added to a manually triggered regression workflow, but never before the calibration requirements in Section 10 pass.

## 15. Error Handling and Secret Safety

- Provider errors are classified as retryable rate limit/server/network failures or non-retryable authentication/request failures.
- `Retry-After` is honored when present; otherwise capped exponential backoff with jitter is used.
- Each case failure is isolated and persisted without aborting unrelated completed work.
- Authentication failures stop the experiment because retries cannot repair them.
- API keys are read only from environment variables and are never stored in configs, SQLite, reports, logs, or exceptions.
- Provider response bodies are truncated and sanitized before error persistence.
- `.env`, live response caches, embedding caches, SQLite databases, and generated reports are Git-ignored unless a specific redacted artifact is intentionally committed.

## 16. Testing and Coverage

All behavior changes follow test-first development.

Line coverage of at least 90% applies only to pure logic modules:

- `domain`;
- `config`;
- `metrics`;
- `budget` logic within `experiments`;
- baseline/comparison logic within `experiments`.

No line-coverage threshold applies to provider networking, CLI glue, optional web routes, templates, or SDK compatibility code. These are verified with:

- fake-provider integration tests;
- temporary-SQLite lifecycle tests;
- CLI smoke tests;
- HTML/JSON report assertions;
- retry/error classification tests at the HTTP boundary;
- one end-to-end offline experiment.

The required M1 GitHub Actions workflow uses Python 3.11, runs Ruff, focused mypy checks, pytest, and the deterministic regression gate. A multi-version matrix belongs to M3.

## 17. Reports and Evidence

M1 exports HTML and JSON containing:

- experiment identity and full configuration;
- retrieval, generation, abstention, system, and cost summaries;
- per-category breakdowns;
- baseline deltas;
- failure cases;
- per-case retrieved evidence and citations;
- judge reasons and calibration status;
- an explicit label distinguishing mock, pilot, and final benchmark runs.

README claims, screenshots, interview answers, and resume bullets must cite a committed redacted artifact or an exact command that reproduces the claim. Mock-run metrics may demonstrate software behavior but may not be described as model-quality results.

## 18. Completion Evidence

M1 is proven only when:

- a clean environment installs from documented commands;
- the zero-key offline command completes end to end;
- the 48-case dataset passes validation;
- retrieval and generation failures are reported separately;
- SQLite stores a completed experiment and supports baseline comparison;
- HTML and JSON reports render from stored records;
- GitHub Actions, tests, lint, type checks, and deterministic regression gates pass.

M2 is proven only when:

- a live preflight records current official price sources and remains under budget;
- a real benchmark completes with actual usage and cost;
- 12 human labels and judge agreement statistics are present;
- position-sensitive judge cases are reported;
- real artifacts support every numeric README and resume claim;
- interview Q&A accurately describes limitations and trade-offs.

M3 is optional and is not part of the completion claim.

## 19. Authoritative References

- DeepSeek official pricing: <https://api-docs.deepseek.com/zh-cn/quick_start/pricing/>
- SiliconFlow pricing: <https://siliconflow.cn/pricing>
- SiliconFlow free-model rate limits: <https://docs.siliconflow.cn/cn/userguide/rate-limits/rate-limit-and-upgradation>
- LangSmith evaluation workflow: <https://docs.langchain.com/langsmith/evaluation>
- DeepEval CI evaluation workflow: <https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd>
- MT-Bench judge bias paper: <https://arxiv.org/abs/2306.05685>
