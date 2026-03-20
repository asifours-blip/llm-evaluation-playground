"""报告生成器单元测试"""
import pytest

from src.reporter.reporter import Reporter


@pytest.fixture
def eval_output():
    """构造一份最小评测输出"""
    return {
        "meta": {
            "model": "fake-model",
            "strategies": ["zero_shot", "cot"],
            "sample_count": 2,
            "total_calls": 4,
            "fail_count": 1,
        },
        "results": [
            {
                "id": 1, "strategy": "zero_shot", "category": "常识",
                "question": "水的化学式？", "gold_answer": "H₂O",
                "response": "H2O", "latency": 0.5, "usage": {"total_tokens": 10},
                "scores": {"semantic_similarity": 0.8, "llm_judge": 4},
            },
            {
                "id": 2, "strategy": "zero_shot", "category": "常识",
                "question": "首都？", "gold_answer": "北京",
                "response": "北京", "latency": 0.6, "usage": {"total_tokens": 8},
                "scores": {"semantic_similarity": 1.0, "llm_judge": 5},
            },
            {
                "id": 1, "strategy": "cot", "category": "常识",
                "question": "水的化学式？", "gold_answer": "H₂O",
                "response": "让我想想…答案是 H2O", "latency": 1.2, "usage": {"total_tokens": 20},
                "scores": {"semantic_similarity": 0.7, "llm_judge": 4},
            },
        ],
        "failures": [{"id": 2, "strategy": "cot", "error": "timeout"}],
    }


class TestGenerate:
    def test_generates_html_file(self, eval_output, tmp_path):
        """生成 HTML 文件并返回路径"""
        out = tmp_path / "report.html"
        path = Reporter().generate(eval_output, output_path=out)
        assert out.exists()
        assert path.endswith("report.html")

    def test_html_contains_strategy_table(self, eval_output, tmp_path):
        """HTML 含策略对比表头和策略名"""
        out = tmp_path / "report.html"
        Reporter().generate(eval_output, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "策略对比" in html
        assert "zero_shot" in html and "cot" in html

    def test_html_contains_detail_and_failures(self, eval_output, tmp_path):
        """HTML 含详细结果和失败案例"""
        out = tmp_path / "report.html"
        Reporter().generate(eval_output, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert "详细结果" in html
        assert "失败案例" in html
        assert "timeout" in html

    def test_html_contains_bar_visualization(self, eval_output, tmp_path):
        """有相似度时生成柱状条"""
        out = tmp_path / "report.html"
        Reporter().generate(eval_output, output_path=out)
        html = out.read_text(encoding="utf-8")
        assert 'class="bar"' in html


class TestSummarize:
    def test_summary_aggregates_by_strategy(self, eval_output):
        """按策略聚合，算平均相似度/延迟"""
        summary = Reporter()._summarize(eval_output["results"])
        by_strat = {s["strategy"]: s for s in summary}

        assert by_strat["zero_shot"]["count"] == 2
        # (0.8 + 1.0) / 2 = 0.9
        assert by_strat["zero_shot"]["avg_similarity"] == 0.9
        # (0.5 + 0.6) / 2 = 0.55
        assert by_strat["zero_shot"]["avg_latency"] == 0.55
        assert by_strat["zero_shot"]["avg_judge"] == 4.5

        assert by_strat["cot"]["count"] == 1

    def test_summary_handles_missing_scores(self):
        """没有 scores 的结果，相似度/Judge 为 None"""
        results = [{"strategy": "zero_shot", "latency": 0.3}]
        summary = Reporter()._summarize(results)
        assert summary[0]["avg_similarity"] is None
        assert summary[0]["avg_judge"] is None
        assert summary[0]["avg_latency"] == 0.3
