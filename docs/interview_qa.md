# 面试 Q&A 预备

针对「LLM 应用评测与快速原型工具」项目可能被追问的问题，及回答要点。

## Q1：为什么要做这个工具？

**要点**：学习 Prompt Engineering 时，需调多个大模型对比 Prompt 效果，手动逐个调用、人工看答案太低效，做了自动化框架。一次配置就能跑三种策略 × N 个样本，自动出量化报告。

## Q2：三种 Prompt 策略的区别是什么？

- **Zero-shot**：不给示例，直接问。测模型零样本能力。
- **Few-shot**：给几个示例，让模型照着风格答。适合格式/风格敏感的任务。
- **Chain-of-Thought (CoT)**：要求模型「一步步思考再给结论」。适合推理类任务，通常能提升正确率。

## Q3：为什么用 OpenAI 兼容接口？

一套代码兼容 DeepSeek / Qwen / GPT / 硅基流动等多种模型，A/B 对比时只改 `model` 参数，不用改调用逻辑。是行业事实标准。

## Q4：两个评分指标是怎么设计的？为什么要两个？

- **语义相似度**：模型回答和参考答案的 embedding 余弦相似度（0-1）。能容忍表述差异（「北京」vs「中国的首都是北京」仍得高分），防「答非所问」。
- **LLM-as-Judge**：让强模型对回答质量打 1-5 分。捕捉相似度反映不了的逻辑正确性（字面像但推理错的情况）。

两指标互补：单用相似度会被「抄参考答案的变形」骗过；单用 Judge 不稳定且贵。组合更可靠。

## Q5：怎么保证测试不烧钱？

`tests/conftest.py` 和各测试文件里用 `unittest.mock.Mock` 替换了模型客户端和 embedding 函数：
- `MockLLMClient.chat` 返回构造好的假回答
- `fake_embed` 返回 md5 生成的确定性向量

所以 38 个测试跑一次零成本，覆盖率 97%。真实评测才用真 Key。

## Q6：重试机制怎么处理的？

`LLMClient.chat` 里指数退避（1s→2s→4s）：
- 429 限流 / 5xx 服务端错误 / 网络抖动 → 重试
- 4xx（除 429，如 400 参数错、401 鉴权失败）→ 不重试直接抛（重试无意义）
- 重试耗尽 → 抛 RuntimeError

## Q7：RAG 是怎么实现的？

LangChain 的 `VectorStoreIndexCreator` 一站式：加载文档 → `RecursiveCharacterTextSplitter` 切片 → `OpenAIEmbeddings` 向量化 → 存进 ChromaDB。查询时用 retriever 取 top_k 片段，拼进 Prompt 让 LLM 结合上下文回答。

## Q8：并发是怎么做的？为什么用线程池而不是异步？

`ThreadPoolExecutor`，默认 4 并发。模型调用是 IO 密集（等 API 响应），线程池在 Python 里足够释放 GIL 的等待时间。异步（asyncio）要改造整个调用链，对这个规模的评测收益不大、复杂度高。

## Q9：用 AI 编程助手（Copilot/Cursor）做了什么？

- 生成 Prompt 模板和测试用例的脚手架
- 重构 Evaluator 的并发逻辑
- 写 Jinja2 报告模板的 CSS

核心逻辑自己设计，AI 负责重复性代码和样板，效率提升明显。

## Q10：项目还有什么不足 / 下一步？

- 数据集只有 20 条、5 个类别，规模偏小，结论统计意义有限
- LLM-as-Judge 没做多 Judge 投票，单 Judge 有偏差
- 没做成本统计（每次策略对比花了多少 token / 钱）
- RAG 还停留在单文档，没做多文档混合检索
