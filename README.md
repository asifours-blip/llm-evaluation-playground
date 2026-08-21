# RAG Quality Lab

一个本地优先、可复现、预算受控的 RAG 评测平台。它把检索、生成、拒答和系统质量拆开度量，把每次实验的代码版本、数据集哈希、Prompt 哈希、随机种子、配置、成本和逐题结果写入 SQLite，并导出可审计的 JSON/HTML 报告。

这不是“接一个模型就算完成”的问答 Demo。项目的目标是回答三个更难的问题：哪种切片与 `top_k` 组合真的改善了检索；模型何时应该拒答；简历里的每个数字能否追溯到实验产物。

## 已验证状态

M1 离线闭环已完成。当前提交附带一份由 commit `57ae92eb0e8953f8fbdc0785184294373c0905d3`、干净工作树生成的真实离线产物：

| 证据 | 结果 | 含义 |
|---|---:|---|
| 知识库 / 数据集 | 12 篇 / 48 题 | 含单文档、多文档、显式域外和“看似相关但无证据”四类场景 |
| 检索配置 | 8 组 | `chunk_size × top_k × prompt_variant`，共 384 个 case-arm 结果 |
| 运行失败 | 0 | 证明离线执行、持久化和报告链路闭环 |
| Recall@k / MRR / context hit | 0.4722 / 0.4740 / 0.4028 | 确定性哈希 embedding 是弱基线，结果没有被包装成高质量检索 |
| 生成 / 拒答 | 1.0 / 1.0 | Mock 回放参考答案，仅证明指标和拒答链路，不代表真实模型质量 |
| 成本 | ¥0 | 离线运行不读取 API Key、不发网络请求 |
| 纯逻辑覆盖率 | 97.49% | 领域、配置、指标、预算、回归模块；网络胶水不靠 mock 覆盖率撑门面 |

可直接检查 [离线 HTML 报告](docs/artifacts/offline-report.html) 和 [规范化 JSON 产物](docs/artifacts/offline-summary.json)。JSON 与 HTML 的 SHA-256 分别为 `b0e0746f4fcf71fda8545209f1686e74115c33c82701ede638c5af7adf62adeb`、`7a63ac4d0fee402cf48a1a1c81d69614e1e62e53519cd51b384d862f93e77a40`。

M2 尚未伪装成完成：仓库没有提交付费 live benchmark，也没有人类填写的 12 条盲标，因此没有发布“模型质量提升”或“Judge 与人工一致率”数字。相关命令、预算闸门和校准逻辑已经具备，外部证据完成后才能升级简历结论。

## 架构与关键取舍

```text
versioned config + dataset + Markdown corpus
                    |
        deterministic chunking / brute-force cosine retrieval
                    |
    structured answer provider (fake or OpenAI-compatible)
                    |
 retrieval metrics | answer metrics | abstention metrics | cost/latency
                    |
       SQLite (WAL) + canonical JSON + self-contained HTML
                    |
           compare + deterministic regression gates
```

- 12 篇文档直接内存暴力余弦检索，不引入向量数据库和部署复杂度。
- `recall@k`、MRR、context hit 与答案 F1/语义相似度分别汇总，避免生成模型掩盖检索失败。
- 无答案题单独统计 abstention accuracy、false-answer rate 和 over-abstention rate。
- Live 运行先按官方价格、调用数和 token 上限计算 1.25× 安全缓冲；未显式确认或超过 90% 预算阈值时不会发请求。
- Runner 只在线程池执行 provider 工作，SQLite 由主线程单写；WAL 与 5 秒 busy timeout 支持报告读取。
- LLM Judge 使用结构化 1–5 分契约和双顺序 pairwise 辅助函数。少于 12 条人工盲标，或一致性不达标时，Judge 指标不得阻断 CI。

模块边界和数据流见 [架构说明](docs/architecture.md)，方案取舍见 [设计规格](docs/superpowers/specs/2026-08-21-rag-quality-lab-design.md)。

## 快速开始

需要 Python 3.11+：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python -m rag_quality_lab.cli validate --config configs/offline.yaml
python -m rag_quality_lab.cli run --config configs/offline.yaml
```

兼容旧演示入口仍可显式运行离线模式，但不再保留第二套评测逻辑：

```bash
python examples/run_eval.py --mock
```

运行会创建被 Git 忽略的 `.ragql/experiments.sqlite3` 和 `artifacts/`。CLI 输出实验 ID、报告绝对路径与摘要，可随后重建或比较：

```bash
rag-quality report --database .ragql/experiments.sqlite3 --experiment EXPERIMENT_ID --output artifacts/rebuilt
rag-quality compare --database .ragql/experiments.sqlite3 --baseline BASELINE_ID --candidate CANDIDATE_ID
rag-quality regression --fixture tests/fixtures/offline_baseline.json
```

## Live 预算预检

示例配置使用 2026-08-21 核验的 DeepSeek 高峰单价证据。先设置环境变量，但不要把 Key 写进 YAML 或提交到仓库：

```powershell
$env:DEEPSEEK_API_KEY = "your-secret"
rag-quality run --config configs/live-deepseek.example.yaml --preflight-only
rag-quality run --config configs/live-deepseek.example.yaml --confirm-live-run
```

预检不读取 API Key、不发网络请求。示例的 96 次生成调用按每次 2,500 输入、512 输出 token 上限估算：未缓冲成本 `¥1.162368`、1.25× 缓冲成本 `¥1.452960`，低于 `¥18` 预检阈值和 `¥20` 硬上限。价格会变化，真正运行前必须从[官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)重新核验并新增日期化价格文件；历史证据不覆盖。

## 质量门禁

GitHub Actions 固定 Python 3.11，执行与本地相同的离线门禁：

```bash
ruff check .
mypy src/rag_quality_lab/domain src/rag_quality_lab/config src/rag_quality_lab/metrics src/rag_quality_lab/experiments/budget.py src/rag_quality_lab/experiments/compare.py
pytest -m "not live" --cov=rag_quality_lab.domain --cov=rag_quality_lab.config --cov=rag_quality_lab.metrics --cov=rag_quality_lab.experiments.budget --cov=rag_quality_lab.experiments.compare --cov-report=term-missing --cov-fail-under=90
rag-quality regression --fixture tests/fixtures/offline_baseline.json
```

## 局限

- 哈希 embedding 故意只作为便宜、可复现的检索弱基线；不能代表生产 embedding。
- 当前公开产物是 Mock，答案分数不可用于比较真实 LLM。
- 48 题适合回归与面试讲解，不足以形成广泛统计结论。
- Judge 的提示、解析、防位置偏差和校准阈值有单测，但没有 live Judge + 人工盲标产物，因此 Judge 指标仍是待验证能力。
- SQLite 适合本地单写实验；高吞吐多写场景应迁移到服务型数据库。

可核验的简历表述与禁用表述见 [简历证据清单](docs/resume_bullets.md)，面试追问见 [Interview Q&A](docs/interview_qa.md)。
