# LLM 应用评测与快速原型工具

用 Python 搭建的 LLM 评测框架：封装统一的 OpenAI 兼容模型调用层，对 **Zero-shot / Few-shot / Chain-of-Thought** 三种 Prompt 策略做自动化对比，输出量化评测报告（语义相似度 + LLM-as-Judge 双指标），并基于 LangChain 探索 RAG 文档问答。

## 功能特性

- **统一模型调用层**：OpenAI 兼容接口，一行配置切换 DeepSeek / Qwen / GPT，内置指数退避重试
- **三策略自动对比**：Jinja2 模板化 Prompt，并发评测，一次跑出三策略效果
- **双指标量化评分**：语义相似度（embedding 余弦）+ LLM-as-Judge（强模型打分 1-5）
- **HTML 评测报告**：策略对比表 + 柱状可视化 + 逐条问答明细
- **RAG 探索**：LangChain 文档切片 → 向量检索 → 上下文增强问答
- **不烧钱的测试**：全量 Mock，38 个单测覆盖核心逻辑，覆盖率 97%

## 技术栈

| 用途 | 技术 |
|------|------|
| 模型调用 | Python + requests，OpenAI 兼容接口 |
| Prompt 模板 / 报告模板 | Jinja2 |
| 评分指标 | OpenAI embeddings（余弦相似度）+ LLM-as-Judge |
| 并发评测 | concurrent.futures 线程池 |
| RAG | LangChain + ChromaDB |
| 测试 | pytest + pytest-cov |

## 快速开始

```bash
# 1. 装依赖
python -m venv venv
source venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt

# 2. 配置 API Key（硅基流动/DeepSeek 等兼容接口均可）
cp .env.example .env
#   编辑 .env 填入 OPENAI_API_KEY

# 3. Mock 模式验证闭环（无需 Key，生成示例报告）
python examples/run_eval.py --mock

# 4. 真实评测（调用大模型，生成正式报告）
python examples/run_eval.py
#   报告输出：data/reports/report.html

# 5. RAG 文档问答（需先装 langchain 依赖）
python examples/rag_demo.py
```

## 项目结构

```
llm-evaluation-playground/
├── src/
│   ├── client/llm_client.py      # 统一模型调用层（重试内置）
│   ├── prompt/                   # Prompt 策略引擎 + 3 个 Jinja2 模板
│   ├── evaluator/evaluator.py    # 并发评测执行器
│   ├── evaluator/scorer.py       # 评分系统（相似度 + LLM-Judge）
│   ├── reporter/reporter.py      # HTML 报告生成
│   └── rag/rag_module.py         # RAG 文档问答
├── examples/run_eval.py          # 评测入口（Mock / 真实）
├── data/qa_samples.json          # 20 条评测数据集
├── tests/                        # 38 个单测，覆盖率 97%
└── requirements.txt
```

## 测试

```bash
pytest tests/ --cov=src --cov-report=term-missing
# 38 passed, 覆盖率 97%
```

测试全程 Mock 模型 API 和 embedding，不消耗真实额度。
