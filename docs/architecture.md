# RAG Quality Lab Architecture

## 目标与边界

RAG Quality Lab 是单机评测系统，不是在线问答服务。它优化的是实验可信度：同一份配置能够重跑，检索与生成失败能够分辨，成本在请求发出前能够阻断，报告中的结论能够回到逐题记录。

当前 M1 支持确定性离线闭环和 OpenAI-compatible live 生成。FastAPI 仪表盘、向量数据库、Docker 和多版本 CI 属于主动降级的 M3 展示项，不是闭环的必要条件。

## 数据流

```text
ExperimentConfig ─┬─ provider / retrieval matrix / budget
                  ├─ versioned EvaluationDataset (48 cases)
                  └─ Markdown knowledge base (12 documents)
                               |
                    stable document and chunk IDs
                               |
              deterministic chunking + embedding cache
                               |
                 in-memory brute-force cosine search
                               |
              StructuredAnswer(answer/citations/abstained)
                               |
       ┌───────────────────────┼────────────────────────┐
       |                       |                        |
 retrieval metrics       generation metrics      abstention metrics
 Recall@k / MRR / hit     EM / bilingual F1 /     accuracy / F1 /
                         semantic similarity       false answer / over
       └───────────────────────┼────────────────────────┘
                               |
             main-thread SQLite writer (WAL, 5 s timeout)
                               |
          canonical JSON + self-contained HTML + SHA-256
                               |
               compare_experiments + regression gate
```

## 模块职责

| 模块 | 职责 | 关键约束 |
|---|---|---|
| `domain` | Pydantic 数据集、配置、结果和生命周期模型 | 跨字段不变量在模型边界拒绝 |
| `config` | UTF-8 JSON/YAML 加载与价格币种验证 | 相对价格路径相对配置文件解析 |
| `retrieval` | Markdown 加载、稳定切片、缓存、暴力余弦检索 | 无外部向量库；相同输入产生相同 chunk ID |
| `providers` | 窄协议、确定性 fake、OpenAI-compatible HTTP | Key 只从环境变量读取；401/403 不重试；错误脱敏 |
| `prompts` | `direct` / `evidence_first` 固定指令与哈希 | Prompt 变更进入实验身份 |
| `metrics` | 检索、答案、拒答、Judge 与人工校准纯函数 | Judge 未校准时不得作为阻断指标 |
| `experiments` | 并发协调、预算预留/结算、SQLite、比较和回归 | Provider 可并发；数据库只由主线程写 |
| `reporting` | 规范化 JSON、自包含 HTML、产物哈希 | Mock 默认 `mock`，live 默认 `pilot`，不可自动冒充 final |
| `cli` | 安全装配上述模块 | Live 需要 `--confirm-live-run`；预检先于 API Key 读取 |

## 为什么不用向量数据库

知识库只有 12 篇文档。这个规模下，内存矩阵与暴力余弦检索足够快，也更容易复现：没有服务进程、索引 schema、网络依赖或持久卷。引入 Chroma、Milvus 或 pgvector 会增加部署面，却不会改善这份评测的核心可信度。

若语料扩大到内存不可接受，或需要高 QPS、多租户和在线增量写入，再把 `InMemoryIndex` 的窄接口替换为向量数据库适配器；指标与实验存储不需要随之重写。

## 可复现身份

每次实验保存：

- Git commit SHA 与 dirty 标志；
- 数据集规范化 SHA-256；
- 两种 Prompt 的 SHA-256；
- 完整配置与随机种子；
- Python 版本；
- 稳定 document/chunk/config/case 标识；
- 每题命中、答案、指标、token、延迟、成本和状态。

JSON 使用排序键和紧凑分隔符序列化。报告另存 SHA-256。由干净工作树产生的产物因此能定位到代码输入，但这不等于不同机器上的网络模型输出会逐字一致；live provider 的非确定性仍需通过多次运行和统计处理。

## 并发与 SQLite

Runner 用有界 `ThreadPoolExecutor` 并发网络/计算工作，最多只保留 `max_workers` 个在途任务。Future 完成后，主线程按稳定键顺序结算预算并写 SQLite，避免多个线程争抢同一连接。

SQLite 开启 WAL、foreign keys 和 5 秒 busy timeout。这个设计允许报告进程读取运行中的库，并减少矩阵跑批与读取之间的锁冲突；它仍是单写架构，不适合高吞吐分布式 worker。

## 预算安全

Live 运行的安全链条如下：

1. 读取日期化价格文件，校验币种、模型与价格新鲜度；
2. 对序列化输入执行保守字节上界、对输出发送 `max_tokens`，并把主调用、一次修复和全部有界重试计入最坏 cache-miss 成本；
3. 乘安全系数，必须低于硬预算的预检比例；
4. 用户显式传入 `--confirm-live-run` 后才构造需要 Key 的 provider；
5. 每个任务调度前原子预留生成与 Judge 的最坏成本，完成后按 provider usage 结算；失败时结算已知 usage，只对已经尝试但 usage 不可得的阶段按预留上限估算；
6. 完整矩阵的缓冲预检不通过时零请求退出；运行中下一次预留可能越过硬上限时停止调度并持久化 `budget_exceeded`。

价格证据是历史快照，不覆盖更新。真实运行前必须从官方来源新增当日价格文件。

## Judge 与人工校准

Judge 辅助模块固定 1–5 分结构化 schema，`score >= 4` 与 `passed=true` 是模型不变量。Pairwise 比较对 A/B 和 B/A 各评一次；若归一化偏好不一致，标为 position-sensitive，不输出赢家。

人工标注导出按可回答性、类别、难度、配置和模型做带种子的轮转分层，物理移除模型、配置、原始 case ID 和 Judge 分数，只保留 opaque sample ID、问题、参考答案、候选答案和 Judge 实际看到的检索证据。SQLite 私下保存 source mapping 与内容哈希，导入时拒绝跨实验或被修改的快照。至少 12 条完整盲标，且 within-one rate ≥ 0.80、MAE ≤ 1.0 时，Judge 才具备回归阻断资格。当前仓库没有真实人工标注产物，所以不能宣称 Judge 已可靠。

Pairwise 不是未接线的 helper：CLI 对匹配的 `(case_id, model)` 输出执行 A/B 与 B/A 两次调用，预算、usage、成本、reason 和 position-sensitive 状态进入 SQLite 与规范化 JSON。位置敏感样本不进入胜率分子。`final` badge 同样不是自由输入，只有 live、完成、clean、零失败且校准达标的实验才允许生成。

## 测试策略

- `domain/config/metrics/budget/compare` 聚焦逻辑执行 branch coverage 门禁，当前为 95.16%；
- provider 用窄 HTTP session fake 验证重试、鉴权、脱敏、结构化修复和 usage 解析；
- runner/store/report/CLI 用集成测试验证真实文件、SQLite、子进程和报告行为；
- CI 排除 `live` marker，不读取 Key、不产生付费请求；
- 固定回归 fixture 指向提交的 384-case 基线产物；CI 会在临时目录完整重跑当前 pipeline 再比较，不能靠嵌入相同的手写 baseline/candidate 或写死 `passed: true`。

## 已知限制与演进条件

- 哈希 embedding 是确定性弱基线，不是生产检索模型；
- M1 Mock 只能证明软件闭环，不能证明 LLM 质量；
- Live 生成与 Judge 执行已支持，但正式 M2 仍需要可用凭据、实际成本产物和独立人工盲标；
- 需要多 worker 写入或远程查询时再迁移 PostgreSQL；
- 需要交互式探索时可增加只读仪表盘，但不能替代规范化报告。
