# 运行、配置与验证详细参考

本文承接提交 [d87a35d](https://github.com/asifours-blip/llm-evaluation-playground/commit/d87a35d18308d74fe5328fd7a7e0d2ce8bc3ca99) 的原首页详细说明。下列历史实验数字和本地运行记录属于各自标注的时间与环境，不代表本次重新执行。当前入口见 [项目首页](../README.md)。

[![CI](https://github.com/asifours-blip/llm-evaluation-playground/actions/workflows/ci.yml/badge.svg)](https://github.com/asifours-blip/llm-evaluation-playground/actions/workflows/ci.yml)

> 仓库名 `llm-evaluation-playground` 为历史链接保留；当前项目名为 **RAG Quality Lab**。

一个本地优先、可复现、预算受控的 RAG 评测平台。它把检索、生成、拒答和系统质量拆开度量，把每次实验的代码版本、数据集哈希、Prompt 哈希、随机种子、配置、成本和逐题结果写入 SQLite，并导出可审计的 JSON/HTML 报告。

这不是“接一个模型就算完成”的问答 Demo。项目的目标是回答三个更难的问题：哪种切片与 `top_k` 组合真的改善了检索；模型何时应该拒答；报告中的每个数字能否追溯到实验产物。

## Repository history

早期提交包含一次本地历史导入；2026-08-21 项目从 LangChain 演示重写为 `rag_quality_lab`。当前行为与质量门禁以 `main` 和 GitHub Actions 为准，不应将早期提交密度理解为线上迭代节奏。

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
| Focused core branch coverage | 95.53% | `domain`、`config`、`metrics`、`experiments.budget`、`experiments.compare` 的 branch coverage；不是全包覆盖率，网络胶水不靠 mock 数字撑门面 |

可直接检查 [离线 HTML 报告](../docs/artifacts/offline-report.html) 和 [规范化 JSON 产物](../docs/artifacts/offline-summary.json)。JSON 与 HTML 的 SHA-256 分别为 `d759dd887e65220123e01d52962a48c96374ca0963647efdcf04abf15231c8e5`、`81385f0ec00dcd77de21a0ae2e0b9f9a31f6cff77518ad404f338ad964162c52`。

M2 主证据已提交，但必须带限定陈述：仓库含 96 case-arm 的 `deepseek-v4-flash` live final、12 条分层人工盲标与通过校准 gate 的摘要（见 `docs/artifacts/final-evidence-summary-2026-08-21.json`）。另有一次 HTTP-instrumented 重跑（`c4f32275-...`，见 `docs/artifacts/http-instrumented-2026-08-21/`）：96 arms 全完成，精确物理 HTTP 总计 **203**，实际成本 ¥0.1735658；本 experiment 的 12 条现场盲标已导入，校准 within-one 100%、MAE ≈ 0.083，**badge=`final`**。历史 final `544dcc6e-...` 的 `http_request_count` 仍为 `null` 且不改写。384-arm 付费全矩阵已完成并升 **final**（`72f6a56d-...`，见 `docs/artifacts/live-384-2026-08-21/`）：384/384、零失败、精确 HTTP **817**、实际成本 ¥0.6644544；同实验 12 条盲标校准 exact/within-one 100%、MAE 0。限定：该 12 条抽到的是易定义题（8 条 abstention 定义 + 4 条 false-answer rate），不得外推为困难样本上的 Judge 可靠性。中断半成品 `3f42f3b6-...` 已标 failed，不作全矩阵证据。

## 架构与关键取舍

```text
versioned config + dataset + Markdown corpus
                    |
        deterministic chunking / cosine or BM25 retrieval
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
- Live 请求在发送前执行输入字节上界和 `max_tokens` 硬限制；预检把主调用、一次结构修复和全部配置重试计入 1.25× 安全缓冲，远程 embedding 的索引、查询、答案三类调用同样进入预检、账本和逐次调用记录。实验记录先于预检创建，未显式确认、计划中任一模型缺少单价或超过 90% 预算阈值时不会发任何请求（包括建索引）。
- Runner 只在线程池执行 provider 工作，SQLite 由主线程单写；WAL 与 5 秒 busy timeout 支持报告读取。
- LLM Judge 使用结构化 1–5 分契约；可执行的 pairwise 命令会调用 A/B 与 B/A、持久化位置敏感结果和成本。少于 12 条人工盲标，或一致性不达标时，Judge 指标不得阻断 CI。

模块边界和数据流见 [架构说明](../docs/architecture.md)，方案取舍见 [设计规格](../docs/design-spec.md)。

## 快速开始

需要 Python 3.11+：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate

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

预检不读取 API Key、不发网络请求。96-arm 示例为每个生成/Judge 阶段预留主调用、一次结构修复和最多 2 次重试，最坏 1,152 次 HTTP；峰值价未缓冲 `¥10.492416`，1.25× 后 `¥13.115520`。8 配置全矩阵示例见 [384 live 配置](../configs/live-deepseek-flash-384.example.yaml)：`max_retries: 0`，预检最坏 1,536 次 HTTP，缓冲后 `¥17.487360`，仍低于 `¥18` 启动阈值和 `¥20` 硬上限。价格会变化，真正运行前必须从[官方价格页](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)重新核验并新增日期化价格文件；历史证据不覆盖。

示例配置的 `fake-hash-*` embedding 在本地计算、不发请求，因此不进入计划，上述金额不含 embedding。若改用远程 embedding 模型，预检按**真实请求批次**计划：每个 dense arm 的索引批次（仅当缓存条目按 provider、模型、chunk 内容与切分方式校验命中时才扣除；`--preflight-only` 不扣除缓存）、每个 dense case-arm 的查询批次、每个 case-arm 的答案相似度批次；BM25 arm 不做索引和查询 embedding，不进入这两类计划。分批规则见下文「Embedding 分批」。价格文件缺少该模型单价时预检直接报错，不会按 0 计费。实际花费按 provider 返回的 usage 结算，usage 缺失或请求失败时按预留上限计入，逐次写入 SQLite `embedding_calls` 表和报告的 `embedding_calls` 字段。

本次零网络结果已固化为 [2026-08-21 live preflight 证据](../docs/artifacts/live-preflight-2026-08-21.json)，SHA-256 为 `56aafe9b0d3a9d68043cf200a9bffcda156d671d2e62172408bd34616770514d`。它证明预算与配置可执行，不是模型质量报告；没有 Key 时绝不能把它改名成 `live-final`。

### 中断恢复与取消

实验记录冻结配置、数据集哈希、语料哈希、prompt 版本和定价快照；每个 case-arm 的预算预留写入 SQLite `ledger_entries`，请求发出前先标记为 dispatched，结果、embedding 记录和结算在同一事务中落库。进程被杀后，下次 `run`/`resume`/`status`/`cancel` 按「主机 + PID + 进程启动时间」识别失去属主的 RUNNING 实验（跨主机时退化为心跳租约超时），将其置为 INTERRUPTED：未发出的预留释放，已发出未结算的调用按预留上限计入并标记为 unknown。

```powershell
rag-quality status --database .ragql/experiments.sqlite3 --experiment EXPERIMENT_ID
rag-quality resume --config configs/live-deepseek.example.yaml --experiment EXPERIMENT_ID --confirm-live-run
rag-quality resume --config configs/live-deepseek.example.yaml --experiment EXPERIMENT_ID --confirm-live-run --retry-unknown
rag-quality cancel --database .ragql/experiments.sqlite3 --experiment EXPERIMENT_ID
```

`resume` 接受 INTERRUPTED 实验；输入与冻结身份不一致时列出变化项并拒绝。已落库的 case-arm 不再请求；unknown 的 case-arm 默认不重发，`--retry-unknown` 重发并另计费用。其余 case-arm 都跑完但仍有 unknown 时，实验结束为 INCOMPLETE 而不是 COMPLETED；INCOMPLETE 只能用 `resume --retry-unknown` 继续，不带该参数会报错并提示，也可以 `cancel` 置为 CANCELLED。状态迁移：RUNNING → COMPLETED / FAILED / BUDGET_EXCEEDED / CANCELLED / INTERRUPTED / INCOMPLETE；INTERRUPTED → RUNNING / CANCELLED；INCOMPLETE →（带 `--retry-unknown`）RUNNING / CANCELLED；其余为终态。账本从已花费金额继续，剩余计划须通过「剩余预算」预检，否则拒绝且状态不变。恢复会产生新花费，所以定价新鲜度按恢复当天判断：冻结的定价快照超过 7 天即拒绝恢复；定价冻结在实验身份里，要用新定价请新建实验。`cancel` 之后执行器不再领取新 case-arm，进行中的 case-arm 在派发下一个阶段（查询/答案 embedding、生成、Judge）前也会停下，不再发出任何新请求；已发出的请求照常结算入账，该 case-arm 记为 `cancelled`（不计分、不进 summary 指标，报告的 `cancelled_cases` 与 summary 的 `cancelled_case_count` 单独计数）。对 INTERRUPTED 实验直接置为 CANCELLED。每次 run/resume 在 `experiment_runs` 记录所用 commit；代码版本变化不阻止恢复，但 `status` 输出（`code_versions`、`warnings`）和报告（`run_attempts`、`code_version_warning`、HTML 顶部警示）会醒目标出结果来自多个代码版本并列出每段运行的 commit。数据库带 `user_version` 版本号：阶段 1 的旧文件打开时自动迁移，更新版本的文件会明确报错。

### Embedding 分批

`provider.embedding_batch_size`（单次最多几条文本）和 `provider.embedding_batch_token_limit`（单次 token 上限，按 UTF-8 字节数 + 每条 8 的保守上界计）控制远程 embedding 分批；都不设时保持旧行为，每个阶段一次请求。分批按输入顺序贪心切分，单条文本超过 token 上限时在预检阶段直接报错，不会发送。计划、预留、结算都以批为单位：账本 `ledger_entries` 的 phase 为 `embedding_index`、`embedding_index#1`……，每批发出前单独标记为 dispatched，`embedding_calls` 每批一条记录（含 `batch_index`/`batch_count`）。某批失败时：之前已成功的批按 usage 照常结算；失败批已发出但无 usage，按预留上限计费（`cost_estimated=true`）；之后未发出的批释放预留、不计费。答案相似度批次按预留上限（生成答案字节上限）而非实际答案长度切分，因此实际请求数恒等于计划请求数。

## 检索基线

每个 retrieval arm 用 `retriever` 选择检索方式，默认 `embedding`（即 `provider.embedding_model`：本地 `fake-hash-*` 哈希弱基线或远程 embedding），`bm25` 为词法基线：

```yaml
retrieval:
  - {chunk_size: 300, chunk_overlap: 50, top_k: 4, prompt_variant: direct}
  - {chunk_size: 300, chunk_overlap: 50, top_k: 4, prompt_variant: direct, retriever: bm25}
```

BM25 自行实现、无新依赖：Okapi BM25，k1=1.2、b=0.75，idf = ln(1 + (N − df + 0.5)/(df + 0.5))（非负）。分词：先 NFKC 归一化并 casefold（全角字母数字等同半角）；连续的 CJK 汉字切成重叠的二字组（单字段落保留单字），不依赖词典或分词模型；其他连续字母/数字为一个 token；标点、空白、下划线作分隔；不做词干化和停用词。查询词按排序后逐项累加，同分按 chunk ID 排序，同样输入永远得到同样结果。BM25 arm 的 config ID 带 `-bm25` 后缀，embedding arm 的 ID 保持不变。BM25 不产生任何外部请求，因此不进入预算计划；答案语义相似度指标仍使用配置的 embedding 模型。

## 数据集版本、标签与留出集

数据集文件的 `version` 是显式版本；内容哈希（`dataset_hash`）覆盖全部字段。每道题可标注：`difficulty`（easy/medium/hard）、`answerability`（answerable/unanswerable）、`review`（`status`: unreviewed/approved/rejected，通过或驳回必须有 `reviewer` 与 `reviewed_at`）、`split`（dev/holdout）。旧数据集文件可直接读取：缺失的难度、复核、切分为空，报告与 `dataset splits` 中显示为「未标注」；未设置的标签不进入哈希，所以现有数据集的 `dataset_hash` 不变。`answerability` 决定指标口径，没有安全的默认值，仍为必填。

```bash
rag-quality dataset splits --dataset data/eval/rag_quality_v1.1.json
rag-quality dataset freeze-holdout --dataset data/eval/rag_quality_v1.1.json
rag-quality dataset verify --dataset data/eval/rag_quality_v1.1.json
```

`rag_quality_v1.1.json`（1.1.0）是在 1.0.0 基础上按固定种子分层抽样得到的切分（dev 32 / holdout 16，holdout 已冻结），并附复核标签；抽样命令、逐题复核结论和局限见 [docs/dataset-v1.1-review.md](../docs/dataset-v1.1-review.md)。1.0.0 文件保持不变，已归档实验仍引用它。

冻结把「数据集名 + 版本」下所有 holdout 题目的完整内容哈希写入同目录的 `<数据集>.holdout-lock.json`（应提交入库，Git 历史即冻结审计记录）。之后该版本的 holdout 有任何改动（改题、把题移入或移出 holdout）都会被 `verify`（退出码 1）和 `validate`/`run`/`resume` 拒绝；只修改 dev 题不影响冻结。要改动 holdout，只能提升 `version` 建立新版本：新版本在冻结前可以运行，但报告会标明 holdout 未冻结、不是留出证据；同一版本不能用不同内容重复冻结。

**流程：用 dev 调参，在冻结的 holdout 上报告。**

1. 给题目标注 `split`，完成人工复核后执行 `freeze-holdout`。
2. 调参阶段在配置里设 `splits: [dev]`，只运行 dev 题；所有对比和取舍只看 dev 结果。
3. 选定最终配置后，用 `splits: [holdout]`（或不设，运行全部）运行一次，报告 holdout 结果。不要根据 holdout 结果回头再调参；若确需再调，应建立新数据集版本并重新冻结新的 holdout。

报告的「Results by dataset split」按 dev / holdout / 未标注分别列出每个配置的指标，从不合并；只有运行时校验通过的冻结 holdout 才标为 held-out 证据，dev 与未标注的结果一律注明「not held-out evidence」。顶部「Quality summary」是全部切分的合并数，不能当作留出验证结果引用。

## 成对比较与单变量约束

```bash
rag-quality compare --database .ragql/experiments.sqlite3 --baseline EXP_A --candidate EXP_B --baseline-config CONFIG_A --candidate-config CONFIG_B --output artifacts/comparison
```

两个配置 arm（可来自同一实验或两个实验）只有在**语料哈希、数据集版本与内容、题目集合（题目 ID 与题干）、随机种子**全部相同时才允许成对比较，否则拒绝并逐项列出不同之处；`pairwise`（LLM Judge 双顺序）在任何计划或请求之前使用同一道门槛。比较报告逐题并排展示两边命中的证据片段、排名与分数，列出仅一侧命中的 chunk、排名变化与指标差，按 dev / holdout / 未标注分别汇总均值差。

比较报告（含实验级 `compare`、`report --baseline` 与 `pairwise` 记录）自动列出两个配置之间所有不同的参数，包括检索参数、provider 参数、prompt 模板哈希与代码版本（commit）；路径、数据库、并发数等不影响测量的字段不计入。不同参数超过一个时，报告顶部显示醒目警告：「本次比较混杂多个变量，差异不能归因于单一改动」。

## Judge 人工盲标

Live 实验会分别保存生成与 Judge 的 model、usage 和合并成本。导出按可回答性、类别、难度、配置和模型做带种子的轮转分层，文件物理移除模型名、配置名、原始 case ID 和 Judge 分数，只暴露 24 位 opaque sample ID。SQLite 私下保存映射与内容哈希，导入时拒绝跨实验或被篡改的样本：

```bash
rag-quality annotate export --database .ragql/experiments.sqlite3 --experiment latest-live --count 12 --output docs/artifacts/human-annotations.jsonl
# 由人工在不知道模型、配置和 Judge 分数的情况下填写 human_score
rag-quality annotate import --database .ragql/experiments.sqlite3 --experiment latest-live --input docs/artifacts/human-annotations.jsonl
rag-quality calibrate --database .ragql/experiments.sqlite3 --experiment latest-live
rag-quality report --database .ragql/experiments.sqlite3 --experiment latest-live --output artifacts/calibrated
```

重建报告会从 SQLite 的人工标注和逐题 Judge 分数确定性重算校准结果。通用 CLI 的阻断基线是至少 12 条、within-one rate >= 0.80 且 MAE <= 1.0；本次真实 benchmark 采用了更严格的人工协议：within-one rate >= 0.90、灾难性分歧为 0、平均有符号差绝对值 <= 0.5。12 条完成盲标、独立复核记录和通过结果分别见 [复核后盲标](../docs/artifacts/blind_label_completed-strict-judge-adjudicated-2026-08-21.csv)、[复核记录](../docs/artifacts/calibration-adjudication-strict-judge-2026-08-21.json) 与 [严格校准报告](../docs/artifacts/calibrate-strict-judge-adjudicated-2026-08-21.json)。

最终配置可以执行双顺序比较；每个 case/model 只有 A/B 与 B/A 归一化偏好一致时才计入胜率：

```bash
rag-quality pairwise --config configs/live-deepseek.example.yaml --database .ragql/experiments.sqlite3 --baseline BASELINE_ID --candidate CANDIDATE_ID --baseline-config CONFIG_A --candidate-config CONFIG_B --output artifacts/pairwise --preflight-only
rag-quality pairwise --config configs/live-deepseek.example.yaml --database .ragql/experiments.sqlite3 --baseline BASELINE_ID --candidate CANDIDATE_ID --baseline-config CONFIG_A --candidate-config CONFIG_B --output artifacts/pairwise --confirm-live-run
```

`report --badge final` 不是自由标签：只有 `live + completed + dirty=false + 零失败 + Judge 校准达标` 才会生成 final 产物。

新版本会把 generation 与 Judge 的物理 HTTP 尝试次数（含重试和结构化输出 repair）逐 case 写入 SQLite，并在报告中仅对完整数据给出精确总数。历史实验无法从 token usage 反推重试次数，因此已归档的 `544dcc6e-b60d-4bd1-bde0-8c8bb89c3508` 仍诚实保留 `http_request_count: null`。commit `d830372` 上的付费重跑 `c4f32275-7d9a-4594-9043-5e61b52d3064` 已消除新跑批次的该限制：精确总计 203 次 HTTP（85 case×2 + 11 case×3），证据见 [HTTP-instrumented 摘要](../docs/artifacts/http-instrumented-2026-08-21/evidence-summary.json)。

## 质量门禁

GitHub Actions 固定 Python 3.11，执行与本地相同的离线门禁：

```bash
ruff check .
mypy --strict src/rag_quality_lab/domain src/rag_quality_lab/config src/rag_quality_lab/retrieval src/rag_quality_lab/metrics src/rag_quality_lab/providers src/rag_quality_lab/experiments src/rag_quality_lab/reporting src/rag_quality_lab/cli.py
pytest -m "not live" --cov=rag_quality_lab.domain --cov=rag_quality_lab.config --cov=rag_quality_lab.metrics --cov=rag_quality_lab.experiments.budget --cov=rag_quality_lab.experiments.compare --cov-report=term-missing --cov-fail-under=90
rag-quality regression --fixture tests/fixtures/offline_baseline.json
```

## 局限

- 哈希 embedding 故意只作为便宜、可复现的检索弱基线；不能代表生产 embedding，也不能把其 answer F1 或 false-answer rate 包装为 RAG 效果优秀。BM25 是第二个本地词法基线，同样不代表生产检索质量；新增基线不改变已归档实验（包括 384-arm live final 约 61.5% 的 false-answer rate）的任何结论。
- `rag_quality_v1.json`（1.0.0）没有切分与复核标签，报告中显示为「未标注」。1.1.0 的 holdout 已冻结，但全部 48 题（含这 16 道 holdout）在切分前都已被历史实验使用过，所以它只对**之后的调参**保持留出，不能当作"未见过的题"的泛化证据；复核由 AI 按仓库所有者要求完成，不是独立人工标注。截至目前，本仓库还没有在 1.1.0 holdout 上跑过的实验结果。
- 离线公开产物仍是 Mock，答案分数不可用于比较真实 LLM。Live 数字必须引用对应 final 报告，且检索仍是哈希 embedding 弱基线。
- 48 题适合回归与示例讲解，不足以形成广泛统计结论。
- Judge 校准是 n=12：96-arm 有区分度，384-arm 偏易定义题；都不能外推为大规模 Judge 可靠性。历史 `544dcc6e` 缺少精确 HTTP 计数；新 final 必须绑定**同一 experiment** 的完整 HTTP 计数与人工校准。
- SQLite 适合本地单写实验；高吞吐多写场景应迁移到服务型数据库。

