"""端到端集成测试：MockLLMClient + PromptEngine + Evaluator + Scorer + Reporter 全链路"""
import hashlib
from pathlib import Path

import pytest

from src.client.llm_client import LLMResponse
from src.prompt.prompt_engine import PromptEngine
from src.evaluator.evaluator import Evaluator, load_dataset
from src.evaluator.scorer import Scorer
from src.reporter.reporter import Reporter


class _MockClient:
    """端到端用的 mock 客户端，同时充当模型和 judge"""

    model = "mock-model"

    def chat(self, messages, **kwargs):
        prompt = messages[0]["content"]
        if "评分标准" in prompt:
            return LLMResponse(content="4", usage={"total_tokens": 5}, latency=0.1)
        question = ""
        for line in prompt.split("\n"):
            if line.startswith("问题："):
                question = line[3:].strip()
                break
        return LLMResponse(content=f"回答：{question[:8]}", usage={"total_tokens": 25}, latency=0.3)


def _fake_embed(texts):
    def vec(t):
        h = hashlib.md5(t.encode()).digest()
        return [b / 255 for b in h[:16]]

    return [vec(t) for t in texts]


@pytest.fixture
def pipeline():
    client = _MockClient()
    scorer = Scorer(embed_fn=_fake_embed, judge_client=_MockClient(), embedding_model="mock", judge_model="mock")
    evaluator = Evaluator(client, PromptEngine(), scorer)
    return evaluator, Reporter()


class TestEndToEnd:
    def test_full_pipeline_generates_report(self, pipeline, tmp_path):
        """完整跑一遍：数据集 → 评测 → 评分 → HTML 报告"""
        evaluator, reporter = pipeline
        dataset = load_dataset("data/qa_samples.json")
        dataset["samples"] = dataset["samples"][:3]  # 3 条够验证

        output = evaluator.evaluate(dataset, strategies=("zero_shot", "few_shot", "cot"))
        assert len(output["results"]) == 9  # 3 × 3
        assert output["meta"]["fail_count"] == 0

        report_path = tmp_path / "report.html"
        path = reporter.generate(output, output_path=report_path)
        assert Path(path).exists()

        html = report_path.read_text(encoding="utf-8")
        # 三策略都进了报告
        for s in ("zero_shot", "few_shot", "cot"):
            assert s in html
        # 每条结果都有评分
        assert all("scores" in r for r in output["results"])
        assert all("semantic_similarity" in r["scores"] for r in output["results"])

    def test_summary_has_three_strategies(self, pipeline):
        """三策略都产出聚合摘要"""
        evaluator, reporter = pipeline
        dataset = load_dataset("data/qa_samples.json")
        dataset["samples"] = dataset["samples"][:2]

        output = evaluator.evaluate(dataset, strategies=("zero_shot", "few_shot", "cot"))
        summary = reporter._summarize(output["results"])
        strategies = {s["strategy"] for s in summary}
        assert strategies == {"zero_shot", "few_shot", "cot"}
        # 每个策略都有相似度和 judge 均值
        for s in summary:
            assert s["avg_similarity"] is not None
            assert s["avg_judge"] is not None
