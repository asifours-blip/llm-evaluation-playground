"""评测执行器单元测试"""
from unittest.mock import Mock

import pytest

from src.client.llm_client import LLMResponse
from src.prompt.prompt_engine import PromptEngine
from src.evaluator.evaluator import Evaluator, load_dataset


@pytest.fixture
def dataset():
    return {
        "version": "1.0",
        "task_description": "答题",
        "few_shot_examples": [{"question": "示例Q", "answer": "示例A"}],
        "samples": [
            {"id": i, "category": "常识", "question": f"问题{i}", "gold_answer": f"答案{i}"}
            for i in range(1, 6)  # 5 条样本
        ],
    }


def make_mock_client(fail_on_id=None):
    """构造 mock 模型客户端，可指定某 id 抛异常"""
    client = Mock()
    client.model = "mock-model"

    def fake_chat(messages, **kwargs):
        prompt = messages[0]["content"]
        # 从 prompt 提取问题里的 id 数字
        for line in prompt.split("\n"):
            if line.startswith("问题：问题"):
                idx = int(line.replace("问题：问题", "").strip())
                if fail_on_id is not None and idx == fail_on_id:
                    raise RuntimeError("模拟调用失败")
        return LLMResponse(content="模拟回答", usage={"total_tokens": 20}, latency=0.4)

    client.chat = Mock(side_effect=fake_chat)
    return client


@pytest.fixture
def evaluator():
    return Evaluator(make_mock_client(), PromptEngine())


class TestEvaluate:
    def test_runs_all_strategy_sample_combos(self, evaluator, dataset):
        """3 策略 × 5 样本 = 15 条结果"""
        out = evaluator.evaluate(dataset, strategies=("zero_shot", "few_shot", "cot"))
        assert len(out["results"]) == 15
        assert out["meta"]["total_calls"] == 15
        assert out["meta"]["fail_count"] == 0

    def test_strategy_subset(self, evaluator, dataset):
        """只跑指定策略"""
        out = evaluator.evaluate(dataset, strategies=("zero_shot",))
        assert len(out["results"]) == 5
        assert all(r["strategy"] == "zero_shot" for r in out["results"])

    def test_failures_collected(self, dataset):
        """单条失败被收进 failures，不中断整体"""
        ev = Evaluator(make_mock_client(fail_on_id=3), PromptEngine())
        out = ev.evaluate(dataset, strategies=("zero_shot",))
        assert len(out["failures"]) == 1
        assert out["failures"][0]["id"] == 3
        assert out["meta"]["fail_count"] == 1
        # 其他 4 条仍成功
        assert len(out["results"]) == 4

    def test_results_sorted_by_strategy_then_id(self, evaluator, dataset):
        """结果按 (策略, id) 排序"""
        out = evaluator.evaluate(dataset, strategies=("cot", "zero_shot"))
        strategies = [r["strategy"] for r in out["results"]]
        # cot 全部在前，zero_shot 在后
        assert strategies[:5] == ["cot"] * 5
        assert strategies[5:] == ["zero_shot"] * 5

    def test_scorer_integrated(self, dataset):
        """传入 scorer 时每条结果带 scores"""
        scorer = Mock()
        scorer.score.return_value = {"semantic_similarity": 0.9}
        ev = Evaluator(make_mock_client(), PromptEngine(), scorer=scorer)
        out = ev.evaluate(dataset, strategies=("zero_shot",))
        assert all("scores" in r for r in out["results"])
        assert out["results"][0]["scores"]["semantic_similarity"] == 0.9

    def test_result_contains_prompt_and_response(self, evaluator, dataset):
        """结果含渲染后的 prompt 和模型 response"""
        out = evaluator.evaluate(dataset, strategies=("zero_shot",))
        r = out["results"][0]
        assert "prompt" in r and "问题1" in r["prompt"]
        assert r["response"] == "模拟回答"
        assert r["latency"] == 0.4


class TestLoadDataset:
    def test_load_dataset(self):
        """加载真实数据集文件"""
        ds = load_dataset("data/qa_samples.json")
        assert ds["version"] == "1.0"
        assert len(ds["samples"]) == 20
        assert "few_shot_examples" in ds
