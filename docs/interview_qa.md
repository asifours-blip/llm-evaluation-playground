# RAG Quality Lab — Interview Q&A

## Q1：为什么不直接展示一个 RAG 问答 Demo？

Demo 只能说明“某次回答看起来能用”，不能定位错误来自检索还是生成，也不能证明换一组参数真的改善了系统。这个项目把评测对象定义成可复现实验：固定数据、Prompt、代码和配置，分别量化检索、答案、拒答、延迟、token、成本和失败。

## Q2：数据集如何避免只测容易的有答案题？

48 题由 24 道单文档、12 道多文档、6 道显式域外和 6 道“主题相关但知识库无证据”的 plausible-unsupported 问题组成。后两类迫使系统区分“不知道”和“凭常识能编”，否则会漏掉 RAG 最危险的假答案模式。

## Q3：为什么把检索与生成指标分开？

答案错可能是没检到、检到了但上下文截断、模型没遵守证据，或者本应拒答却编了。只看最终答案无法归因。项目先算 Recall@k、MRR、reference evidence context-hit，再算 exact match、双语 token F1 和 embedding 相似度；面试时可以用逐题命中解释哪一层失败。

## Q4：无答案题怎么评分？

除通用答案指标外，单独计算 abstention accuracy、precision/recall/F1、false-answer rate 和 over-abstention rate。false answer 是无证据却回答，风险通常高于普通答错；over-abstention 是有证据却拒答，影响可用性。两者不能用一个平均 F1 混掉。

## Q5：为什么 12 篇文档不用向量数据库？

内存暴力余弦检索已经足够快，依赖更少、索引行为更透明，也更容易通过内容哈希复现。引入向量数据库会多出服务、schema、持久卷和版本变量，却不会提升这个规模下的评测可信度。语料或并发达到内存方案瓶颈时，再替换窄 `InMemoryIndex` 接口。

## Q6：当前检索只有约 0.47 Recall@k，是不是项目效果很差？

这是刻意保留的确定性哈希 embedding 弱基线，不是宣传指标。它的价值是让离线 CI 可重复、零成本，并证明报告会如实暴露弱检索，而不是让完美 mock 答案掩盖它。真实模型质量结论必须来自 M2 的生产 embedding/生成模型 benchmark。

## Q7：如何保证实验可复现？

每次运行保存 commit SHA、dirty 标志、数据集哈希、Prompt 哈希、随机种子、Python 版本和完整配置；document、chunk、case 和 retrieval config 都有稳定 ID。逐题结果与报告规范化序列化并计算 SHA-256。对于 live 模型，我会明确“输入可复现”不等于“输出逐字确定”，需要重复实验处理随机性。

## Q8：为什么用 SQLite，如何处理并发锁？

本地实验库的写入规模小，SQLite 易检查、易携带。线程池只做 provider/指标工作，所有 Future 回到主线程后再写同一连接；同时开启 WAL、foreign keys 和 5 秒 busy timeout，降低报告读取与跑批的锁冲突。多机多 writer 才有迁移 PostgreSQL 的必要。

## Q9：预算 20 元如何保证不超？

不是先假设 20 元够，而是读取带核验日期和官方 URL 的价格 YAML，按总调用数 × 每次输入/输出 token 上限计算 cache-miss 最坏成本，再乘 1.25 安全系数；缓冲成本必须低于硬上限的 90%。运行中每次调度前预留上限成本，收到 usage 后结算实际成本，下一次预留可能越线就停止并持久化 `budget_exceeded`。

## Q10：Provider 错误与密钥如何处理？

Key 只按配置的环境变量名读取，不写入配置。401/403 立即失败；429、5xx 和网络错误做有界指数退避；错误文本会替换 Key 和 Bearer token，并限制持久化长度。结构化回答解析失败只允许一次“修复为 JSON”调用，仍失败就记录明确错误，不继续猜字段。

## Q11：LLM-as-Judge 如何减少偏差？

Judge 输出必须满足 1–5 分 schema，`passed` 与分数阈值强绑定。Pairwise 辅助流程用 A/B 和 B/A 两种顺序各评一次；归一化结果不一致就标记 position-sensitive。更关键的是，Judge 不能自证可靠：至少取 12 条盲标与人工比较，within-one rate ≥ 0.80 且 MAE ≤ 1.0 后才允许参与阻断门禁。

## Q12：现在有 Judge 与人工一致率吗？

没有。代码和测试覆盖了结构化 rubric、顺序控制、盲标导入和一致性阈值，但仓库没有真实 live Judge 分数和独立人工标注。面试中应把它说成“已实现但待外部证据验证”，不能说成“Judge 已可靠”。

## Q13：为什么覆盖率只卡纯逻辑模块？

领域、配置、指标、预算和比较逻辑适合穷举边界，当前聚焦 branch coverage 为 97.30%。HTTP、CLI、SQLite 与模板属于胶水边界，用行为集成测试验证。若为了全仓数字给每条网络路径堆 mock，覆盖率会上升，但不会增加真实可靠性。

## Q14：如何避免 CI 回归 fixture 造假？

Fixture 保存完整的 baseline/candidate `ExperimentRecord` 和规则。CLI 必须解析它们，调用生产 `compare_experiments` 与 `evaluate_regression` 再决定退出码；不是读取一个手写的 `passed: true`。因此改变比较方向、缺失指标或越过阈值都会让 CI 失败。

## Q15：Mock 运行到底证明了什么、没证明什么？

它证明 48 题 × 8 配置能完成切片、检索、结构化生成、指标、并发、SQLite、报告和回归闭环，而且不依赖网络。它不证明真实 LLM 的正确性、faithfulness、成本、延迟或 Judge 可靠性。README 和简历证据表把这两层明确分开。

## Q16：如果继续做，优先级是什么？

先完成 M2：核验当日官方价格，预算预检，运行小规模 live pilot，再跑 final 配置；导出 12–15 条盲标交给人类填写，计算 Judge 一致率后才发布模型质量数字。仪表盘、Docker 和三版本 CI 都是展示增强，不能抢在证据闭环前面。
