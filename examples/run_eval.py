"""LLM 评测工具入口脚本

用法（在项目根目录运行）：
  python examples/run_eval.py --mock   # 无 API Key：模拟客户端演示闭环，生成示例报告
  python examples/run_eval.py          # 有 .env 配置 Key：真实调用大模型评测

真实模式默认只跑前 5 个样本 × 3 策略 = 15 次调用，控制 token 消耗。
"""
import hashlib
import os
import sys
from pathlib import Path

# 让脚本能 import 项目根的 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.client.llm_client import LLMClient, LLMResponse
from src.prompt.prompt_engine import PromptEngine
from src.evaluator.evaluator import Evaluator, load_dataset
from src.evaluator.scorer import Scorer
from src.reporter.reporter import Reporter


class MockLLMClient:
    """无 API Key 时的演示客户端：返回模拟回答，验证整套闭环能跑通"""

    model = "mock-model"

    def chat(self, messages, **kwargs):
        prompt = messages[0]["content"]
        # Judge 评分请求：返回一个分数
        if "评分标准" in prompt:
            return LLMResponse(content="4", usage={"total_tokens": 5}, latency=0.1)
        # 普通问答：从 prompt 抽取问题
        question = ""
        for line in prompt.split("\n"):
            if line.startswith("问题："):
                question = line[3:].strip()
                break
        if "一步步" in prompt:  # CoT 策略：模拟推理过程
            content = f"分析：这是关于「{question}」的问题。\n答案：模拟回答"
        else:
            content = f"模拟回答：{question[:12]}"
        return LLMResponse(content=content, usage={"total_tokens": 30}, latency=0.3)


def fake_embed(texts):
    """假 embedding：按文本 md5 生成确定性向量，让相似度有差异、可复现"""

    def vec(t):
        h = hashlib.md5(t.encode()).digest()
        return [b / 255 for b in h[:16]]

    return [vec(t) for t in texts]


def main():
    mock_mode = "--mock" in sys.argv or not os.getenv("OPENAI_API_KEY")

    dataset = load_dataset("data/qa_samples.json")
    # 控制规模：真实模式只跑前 5 条，省 token；Mock 模式跑前 5 条够演示
    sample_limit = 5
    dataset["samples"] = dataset["samples"][:sample_limit]

    if mock_mode:
        print("[Mock 模式] 未配置 API Key，使用模拟客户端演示闭环")
        client = MockLLMClient()
        scorer = Scorer(embed_fn=fake_embed, judge_client=MockLLMClient(), embedding_model="mock", judge_model="mock")
    else:
        print(f"[真实模式] 模型：{os.getenv('DEFAULT_MODEL')}，Judge：{os.getenv('JUDGE_MODEL')}")

        client = LLMClient()
        scorer = Scorer(judge_client=LLMClient())

    engine = PromptEngine()
    evaluator = Evaluator(client, engine, scorer)

    total_calls = len(dataset["samples"]) * 3
    print(f"开始评测：{len(dataset['samples'])} 个样本 × 3 策略 = {total_calls} 次调用\n")

    output = evaluator.evaluate(dataset, strategies=("zero_shot", "few_shot", "cot"))

    reporter = Reporter()
    path = reporter.generate(output, output_path="data/reports/report.html")

    print(f"[完成] 报告已生成：{path}")
    print(f"   成功 {len(output['results'])} 条，失败 {len(output['failures'])} 条\n")
    print("策略对比摘要：")
    print(f"  {'策略':<12} {'平均相似度':<12} {'平均Judge':<10} {'平均延迟(s)':<10}")
    for s in reporter._summarize(output["results"]):
        sim = s["avg_similarity"] if s["avg_similarity"] is not None else "—"
        judge = s["avg_judge"] if s["avg_judge"] is not None else "—"
        lat = s["avg_latency"] if s["avg_latency"] is not None else "—"
        print(f"  {s['strategy']:<12} {str(sim):<12} {str(judge):<10} {str(lat):<10}")


if __name__ == "__main__":
    main()
