# Resume Evidence Ledger

简历只能使用下列“已验证”表述。所有数字都能回到 [离线 JSON](artifacts/offline-summary.json)、[离线 HTML](artifacts/offline-report.html) 或 CI 配置。标为 M2 的模板在真实 live 与人工证据产生前禁止填写。

## 中文：当前可用（M1 软件工程证据）

- 构建本地优先的 RAG 评测平台，将检索、生成、拒答与系统指标拆分度量；在 12 篇文档、48 道版本化问题和 8 组检索配置上完成 384 个确定性 case-arm 运行，零执行失败，并导出可追溯 JSON/HTML 报告。
- 设计可复现实验身份，持久化 commit SHA、工作树状态、数据集/Prompt 哈希、随机种子、完整配置、逐题命中、token、延迟和成本；采用 SQLite WAL 与主线程单写协调并发 runner，避免跑批和报告读取互相阻塞。
- 实现日期化价格证据、全矩阵 token 上限预检、1.25× 安全缓冲、逐请求成本预留/结算和 ¥20 硬预算停止机制；DeepSeek 示例的 96 次请求高峰最坏预检为 ¥1.162368，缓冲后 ¥1.452960。
- 建立 Python 3.11 CI 门禁，结合 Ruff、strict mypy、真实回归 fixture 和聚焦纯逻辑 branch coverage；领域、配置、指标、预算和比较模块覆盖率达到 97.49%，且 CI 不读取 API Key、不调用付费模型。

## English: currently usable (M1 software evidence)

- Built a local-first RAG evaluation platform that separates retrieval, generation, abstention, and system metrics; executed 384 deterministic case-arms across 12 documents, 48 versioned questions, and 8 retrieval configurations with zero pipeline failures and auditable JSON/HTML reports.
- Designed reproducible experiment identities covering commit/dirty state, dataset and prompt hashes, seed, full configuration, per-case retrieval hits, tokens, latency, and cost; coordinated concurrent provider work with a single SQLite WAL writer.
- Implemented dated pricing evidence, full-matrix token-cap preflight, a 1.25× safety buffer, per-call reservation/settlement, and a CNY 20 hard stop; the 96-call DeepSeek example preflights at CNY 1.162368 unbuffered and CNY 1.452960 buffered under peak rates.
- Enforced a Python 3.11 quality gate with Ruff, strict mypy, a real regression fixture, and focused branch coverage; pure domain/config/metric/budget/comparison logic reached 97.49% without paid API calls in CI.

## 不可写成“模型效果”的数字

- `answer_exact_match/F1/semantic_similarity = 1.0`：Mock 直接回放参考答案，只验证评分路径。
- `abstention accuracy/F1 = 1.0`：Mock 按数据集标签设置 `abstained`，只验证拒答路径。
- `Recall@k = 0.4722`、MRR `0.4740`、context hit `0.4028`：这是哈希 embedding 弱基线，不是优化后的生产效果。
- `¥0`：只适用于离线产物，不代表 live benchmark 免费。

## M2 完成后才可填写的模板

以下项目必须从 `live-final` 报告与至少 12 条真实人工盲标中取值，不能估算：

- 在 `[N]` 道题、`[K]` 组候选配置上，以 `[MODEL]` 完成付费 benchmark；相对 baseline 将 Recall@k 从 `[A]` 提升到 `[B]`、答案 F1 从 `[C]` 提升到 `[D]`，实际成本 `[CNY]`、P95 延迟 `[MS]`，完整产物哈希为 `[SHA]`。
- 校准 LLM-as-Judge 与 12–15 条人工盲标，within-one 一致率 `[X%]`、MAE `[Y]`；仅当阈值通过时才能写“Judge 参与回归门禁”，否则写“Judge 仅作诊断指标”。

## 发布前核对

- 报告 badge 必须是 `final`，不能把 `mock` 或 `pilot` 改名；
- 报告 identity 必须有 commit SHA，且 `dirty=false`；
- 成本使用 provider usage 与日期化官方价格，不用余额变化倒推；
- 不发布 API Key、Authorization header 或未经审查的原始响应；
- 简历数字与报告保持同一精度，不挑选单题最好结果冒充整体结果。
