"""评分系统单元测试

embed_fn 注入假实现，judge_client 用 Mock，全程不调真实 API。
"""
from unittest.mock import Mock

import pytest

from src.evaluator.scorer import Scorer
from src.client.llm_client import LLMResponse


def make_scorer(embed_fn=None, judge_client=None):
    return Scorer(embed_fn=embed_fn, judge_client=judge_client, embedding_model="fake", judge_model="fake")


class TestSemanticSimilarity:
    def test_identical_text_similarity_one(self):
        """相同文本 → 1.0"""
        # 假 embedding：相同输入返回相同向量
        def fake_embed(texts):
            vec = [1.0, 0.0, 0.0]
            return [vec[:] for _ in texts]
        scorer = make_scorer(embed_fn=fake_embed)
        assert scorer.semantic_similarity("北京", "北京") == 1.0

    def test_orthogonal_vectors_similarity_zero(self):
        """正交向量 → 0.0"""
        def fake_embed(texts):
            return [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))]
        scorer = make_scorer(embed_fn=fake_embed)
        assert scorer.semantic_similarity("a", "b") == 0.0

    def test_partial_similarity_in_range(self):
        """部分相似 → 0~1 之间"""
        def fake_embed(texts):
            return [[1.0, 1.0] if i == 0 else [1.0, 0.0] for i in range(len(texts))]
        scorer = make_scorer(embed_fn=fake_embed)
        # cos = 1/(sqrt2 * 1) ≈ 0.7071
        sim = scorer.semantic_similarity("a", "b")
        assert 0 < sim < 1
        assert round(sim, 4) == round(1 / (2 ** 0.5), 4)


class TestLLMJudge:
    def test_judge_extracts_digit(self):
        """Judge 返回含数字 → 提取首位"""
        judge = Mock()
        judge.chat.return_value = LLMResponse(content="评分：4 分", usage={}, latency=0.1)
        scorer = make_scorer(embed_fn=lambda t: [[0.0] for _ in t], judge_client=judge)
        assert scorer.llm_judge("q", "pred", "gold") == 4

    def test_judge_no_digit_returns_zero(self):
        """Judge 返回无数字 → 0"""
        judge = Mock()
        judge.chat.return_value = LLMResponse(content="无法判断", usage={}, latency=0.1)
        scorer = make_scorer(embed_fn=lambda t: [[0.0] for _ in t], judge_client=judge)
        assert scorer.llm_judge("q", "pred", "gold") == 0

    def test_judge_passes_zero_temperature(self):
        """Judge 调用使用 temperature=0 保证稳定"""
        judge = Mock()
        judge.chat.return_value = LLMResponse(content="3", usage={}, latency=0.1)
        scorer = make_scorer(embed_fn=lambda t: [[0.0] for _ in t], judge_client=judge)
        scorer.llm_judge("q", "p", "g")
        kwargs = judge.chat.call_args.kwargs
        assert kwargs["temperature"] == 0


class TestScoreCombination:
    def test_score_without_judge(self):
        """无 judge_client → 只有 semantic_similarity"""
        scorer = make_scorer(embed_fn=lambda t: [[1.0, 0.0] for _ in t])
        out = scorer.score("pred", "gold", question="q")
        assert "semantic_similarity" in out
        assert "llm_judge" not in out

    def test_score_with_judge(self):
        """有 judge_client + question → 两指标都有"""
        judge = Mock()
        judge.chat.return_value = LLMResponse(content="5", usage={}, latency=0.1)
        scorer = make_scorer(embed_fn=lambda t: [[1.0, 0.0] for _ in t], judge_client=judge)
        out = scorer.score("pred", "gold", question="q")
        assert "semantic_similarity" in out
        assert out["llm_judge"] == 5
