# Resume Evidence Ledger

简历只能使用下列“已验证”表述。所有数字都能回到 [离线 JSON](artifacts/offline-summary.json)、[离线 HTML](artifacts/offline-report.html) 或 CI 配置。标为 M2 的模板在真实 live 与人工证据产生前禁止填写。

## 中文：当前可用（M1 软件工程证据）

- 构建本地优先的 RAG 评测平台，将检索、生成、拒答与系统指标拆分度量；在 12 篇文档、48 道版本化问题和 8 组检索配置上完成 384 个确定性 case-arm 运行，零执行失败，并导出可追溯 JSON/HTML 报告。
- 设计可复现实验身份，持久化 commit SHA、工作树状态、数据集/Prompt 哈希、随机种子、完整配置、逐题命中、token、延迟和成本；采用 SQLite WAL 与主线程单写协调并发 runner，避免跑批和报告读取互相阻塞。
- 实现日期化价格证据、请求级输入/输出硬上限、修复与重试 headroom、1.25× 安全缓冲和 ¥20 预算闸门；DeepSeek 示例按最坏 1,152 次 HTTP 尝试预检，峰值价 ¥10.492416，缓冲后 ¥13.115520。
- 建立 Python 3.11 CI 门禁，结合 Ruff、全包 strict mypy、真实 384-case pipeline 回归和聚焦 branch coverage；领域、配置、指标、预算和比较模块覆盖率达到 95.53%，且 CI 不读取 API Key、不调用付费模型。

## English: currently usable (M1 software evidence)

- Built a local-first RAG evaluation platform that separates retrieval, generation, abstention, and system metrics; executed 384 deterministic case-arms across 12 documents, 48 versioned questions, and 8 retrieval configurations with zero pipeline failures and auditable JSON/HTML reports.
- Designed reproducible experiment identities covering commit/dirty state, dataset and prompt hashes, seed, full configuration, per-case retrieval hits, tokens, latency, and cost; coordinated concurrent provider work with a single SQLite WAL writer.
- Implemented dated pricing evidence, request-level input/output caps, repair/retry headroom, a 1.25× safety buffer, and a CNY 20 budget gate; the DeepSeek example reserves 1,152 worst-case HTTP attempts at CNY 10.492416 peak, or CNY 13.115520 buffered.
- Enforced a Python 3.11 quality gate with Ruff, package-wide strict mypy, a real 384-case pipeline regression, and 95.53% focused branch coverage without paid API calls in CI.

## 不可写成“模型效果”的数字

- `answer_exact_match/F1/semantic_similarity = 1.0`：Mock 直接回放参考答案，只验证评分路径。
- `abstention accuracy/F1 = 1.0`：Mock 按数据集标签设置 `abstained`，只验证拒答路径。
- `Recall@k = 0.4722`、MRR `0.4740`、context hit `0.4028`：这是哈希 embedding 弱基线，不是优化后的生产效果。
- `¥0`：只适用于离线产物，不代表 live benchmark 免费。

## M2 已验证（须带样本与观测性限定）

数据来源：[final 证据摘要](artifacts/final-evidence-summary-2026-08-21.json) 与 [strict-judge final JSON](artifacts/final-strict-judge/544dcc6e-b60d-4bd1-bde0-8c8bb89c3508.json)。发布时必须保留下列限定，不可删掉样本规模或 HTTP 观测性说明：

- 在 48 道题、2 组检索配置（共 96 case-arm）上，以 `deepseek-v4-flash` 完成付费 live benchmark；零失败，实际成本 ¥0.2820596；检索 Recall@k / MRR / context hit 为 0.6944 / 0.5544 / 0.6111（仍使用本地哈希 embedding，不得写成生产 embedding 效果）；JSON SHA-256 `ab971a6b24dfb2c6f25677b201eee5cb26639225fc14ce8374654a790b602d2f`。
- 对 12 条分层人工盲标做 Judge 校准（1 条边界样本独立复核）：within-one rate 100%、灾难性分歧 0、MAE 0.333、gate pass；n=12 下以容差一致率为主判据，不得外推为大规模 Judge 可靠性证明。
- 该次 final 产物生成于 HTTP 物理尝试计数 instrumentation 之前，报告中 `http_request_count` 诚实为 `null`；新 final 门禁已要求逐 case 精确 HTTP 计数，重跑付费矩阵前不得声称“已消除该观测性限制”。

## 仍不可填写的模板

以下在缺少对应新证据前禁止估算填写：

- 384 case-arm 全矩阵付费 live（当前仅有预检配置与离线 384；未确认预算前不得跑、不得写“已完成 384 live”）。
- 相对某 baseline 的“答案 F1 从 A 提升到 B / P95 延迟”等对比句——须来自同协议 compare 产物，不能从单次 final 摘要手算。

## 发布前核对

- 报告 badge 必须是 `final`，不能把 `mock` 或 `pilot` 改名；
- 报告 identity 必须有 commit SHA，且 `dirty=false`；
- 成本使用 provider usage 与日期化官方价格，不用余额变化倒推；
- 不发布 API Key、Authorization header 或未经审查的原始响应；
- 简历数字与报告保持同一精度，不挑选单题最好结果冒充整体结果。
