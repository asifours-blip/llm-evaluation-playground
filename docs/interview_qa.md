# RAG Quality Lab — Interview Q&A

## Q1：为什么不直接展示一个 RAG 问答 Demo？

Demo 只能说明“某次回答看起来能用”，不能定位错误来自检索还是生成，也不能证明换一组参数真的改善了系统。这个项目把评测对象定义成可复现实验：固定数据、Prompt、代码和配置，分别量化检索、答案、拒答、延迟、token、成本和失败。

## Q2：数据集如何避免只测容易的有答案题？

48 题由 24 道单文档、12 道多文档、6 道显式域外和 6 道“主题相关但知识库无证据”的 plausible-unsupported 问题组成。后两类迫使系统区分“不知道”和“凭常识能编”，否则会漏掉 RAG 最危险的假答案模式。

## Q3：为什么把检索与生成指标分开？

答案错可能是没检到、检到了但上下文截断、模型没遵守证据，或者本应拒答却编了。只看最终答案无法归因。项目先算 Recall@k、MRR、reference evidence context-hit，再算 exact match、双语 token F1 和 embedding 相似度；面试时可以用逐题命中解释哪一层失败。

## Q4：无答案题怎么评分？

除通用答案指标外，单独计算 abstention accuracy、precision/recall/F1、false-answer rate 和 over-abstention rate。有效拒答必须同时满足结构化 `abstained=true` 和正文明确表达证据不足，不能靠一个布尔位掩盖实质性回答；答案 F1/EM 只在可回答题上汇总。false answer 与 over-abstention 也不能用一个平均 F1 混掉。

## Q5：为什么 12 篇文档不用向量数据库？

内存暴力余弦检索已经足够快，依赖更少、索引行为更透明，也更容易通过内容哈希复现。引入向量数据库会多出服务、schema、持久卷和版本变量，却不会提升这个规模下的评测可信度。语料或并发达到内存方案瓶颈时，再替换窄 `InMemoryIndex` 接口。

## Q6：当前检索只有约 0.47 Recall@k，是不是项目效果很差？

这是刻意保留的确定性哈希 embedding 弱基线，不是宣传指标。它的价值是让离线 CI 可重复、零成本，并证明报告会如实暴露弱检索，而不是让完美 mock 答案掩盖它。真实模型质量结论必须来自 M2 的生产 embedding/生成模型 benchmark。

## Q7：如何保证实验可复现？

每次运行保存 commit SHA、dirty 标志、数据集哈希、Prompt 哈希、随机种子、Python/关键依赖版本、生成参数和完整配置；embedding cache key 还包含 provider/endpoint identity。document、chunk、case 和 retrieval config 都有稳定 ID，报告规范化序列化并计算 SHA-256。对于 live 模型，我会明确“输入可复现”不等于“输出逐字确定”。

## Q8：为什么用 SQLite，如何处理并发锁？

本地实验库的写入规模小，SQLite 易检查、易携带。线程池只做 provider/指标工作，所有 Future 回到主线程后再写同一连接；同时开启 WAL、foreign keys 和 5 秒 busy timeout，降低报告读取与跑批的锁冲突。多机多 writer 才有迁移 PostgreSQL 的必要。

## Q9：预算 20 元如何保证不超？

不是先假设 20 元够，而是读取带核验日期和官方 URL 的价格 YAML。请求发送前会截断序列化输入并设置 `max_tokens`；预检把主调用、一次结构修复和 `max_retries + 1` 次尝试全部计入 cache-miss 最坏成本，再乘 1.25 安全系数。完整矩阵缓冲成本超过 90% 阈值时零请求退出，运行中则先预留、后按 usage 结算。

## Q10：Provider 错误与密钥如何处理？

Key 只按配置的环境变量名读取，不写入配置。401/403 不重试；429、5xx 和网络错误做有界指数退避；错误文本会替换 Key 和 Bearer token，并限制持久化长度。检索、生成、指标和 Judge 异常按阶段落成失败 case，其他题继续；已知 usage 按实结算，已尝试但 usage 不可得的阶段按预留上限保守记账。结构化解析只允许一次 JSON 修复。

## Q11：LLM-as-Judge 如何减少偏差？

Judge 输出必须满足 1–5 分 schema，`passed` 与分数阈值强绑定。可执行的 pairwise 命令用 A/B 和 B/A 两种顺序各评一次，将 usage、成本、reason 与 position-sensitive 结果持久化；归一化不一致的样本不计入胜率。更关键的是，Judge 不能自证可靠：至少 12 条分层 opaque 盲标达到 within-one ≥ 0.80、MAE ≤ 1.0 后才允许参与阻断门禁。

## Q12：现在有 Judge 与人工一致率吗？

有，但是小样本、带限定。`docs/artifacts/final-evidence-summary-2026-08-21.json` 记录了 12 条分层盲标（含 1 条边界复核）：within-one rate 100%、灾难性分歧 0、MAE ≈ 0.33，校准 gate 通过。面试应说“在 n=12 协议下通过了更严的人工校准门禁”，不能说成“Judge 已在大规模上可靠”或省略样本规模。

## Q13：为什么覆盖率只卡纯逻辑模块？

领域、配置、指标、预算和比较逻辑适合穷举边界，当前聚焦 branch coverage 为 95.53%。Provider、CLI、SQLite 与模板边界用行为集成测试验证；全包 strict mypy 另行覆盖类型边界。为了全仓数字给每条网络路径堆 mock，不会增加真实可靠性。

## Q14：如何避免 CI 回归 fixture 造假？

Fixture 只保存真实配置、提交的 384-case baseline 报告路径和规则。CI 在临时目录完整重跑当前 48×8 pipeline，再按 `(case_id, config_id, model)` 比较；检索、runner、指标或 prompt 退化都会改变候选产物。它不是把一条手写 baseline/candidate 复制两遍，更不是读取 `passed: true`。

## Q15：Mock 运行到底证明了什么、没证明什么？

它证明 48 题 × 8 配置能完成切片、检索、结构化生成、指标、并发、SQLite、报告和回归闭环，而且不依赖网络。它不证明真实 LLM 的正确性、faithfulness、成本、延迟或 Judge 可靠性。README 和简历证据表把这两层明确分开。

## Q16：如果继续做，优先级是什么？

M2 主证据已具备（96-arm live + 12 条盲标校准）。下一步优先：① 在确认预算后，用已 instrument 的 HTTP 计数重跑 live，消掉历史 `http_request_count: null`；② 若要写 384-arm 付费结论，先跑 dated preflight 并显式确认；③ 需要对比句时用 `compare`/`pairwise` 产物，不手算。仪表盘、Docker 和多版本 CI 仍是展示增强，不能抢在新证据前面。
