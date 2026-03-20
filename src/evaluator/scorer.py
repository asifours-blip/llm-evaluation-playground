"""评分系统：语义相似度 + LLM-as-Judge 双指标

- semantic_similarity：把「模型回答」与「参考答案」各自做 embedding，算余弦相似度（0~1）。
  越接近 1 说明语义越接近参考答案，能容忍表述差异（如「北京」vs「中国的首都是北京」）。
- llm_judge：让一个更强的模型对回答质量打 1~5 分，捕捉相似度无法反映的逻辑正确性。

两指标互补：相似度防「答非所问」，Judge 防「字面像但逻辑错」。
"""
import math
import os
from typing import Callable


class Scorer:
    def __init__(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        judge_client=None,
        embedding_model: str | None = None,
        judge_model: str | None = None,
    ):
        # embed_fn 可注入：测试时传假函数，生产时留空走默认 OpenAI embeddings
        self._embed_fn = embed_fn
        self.judge_client = judge_client  # LLMClient 实例，None 则不做 Judge 评分
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        self.judge_model = judge_model or os.getenv("JUDGE_MODEL")

    def _default_embed(self, texts: list[str]) -> list[list[float]]:
        """默认 embedding 实现：走 OpenAI 兼容接口（硅基流动等）"""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1"),
        )
        resp = client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return (self._embed_fn or self._default_embed)(texts)

    def semantic_similarity(self, prediction: str, gold_answer: str) -> float:
        """两个文本的 embedding 余弦相似度，0~1"""
        v1, v2 = self.embed([prediction, gold_answer])
        return round(self._cosine(v1, v2), 4)

    def llm_judge(self, question: str, prediction: str, gold_answer: str) -> int:
        """LLM-as-Judge：返回 1~5 整数分"""
        prompt = (
            "请评估以下 AI 回答的质量，只返回 1 到 5 的整数分数，不要任何其他内容。\n\n"
            f"问题：{question}\n参考答案：{gold_answer}\nAI 回答：{prediction}\n\n"
            "评分标准：5=完全正确，4=基本正确，3=部分正确，2=大部分错误，1=完全错误。\n分数："
        )
        resp = self.judge_client.chat(
            [{"role": "user", "content": prompt}],
            model=self.judge_model,
            temperature=0,
        )
        digits = "".join(ch for ch in resp.content if ch.isdigit())
        return int(digits[0]) if digits else 0

    def score(self, prediction: str, gold_answer: str, question: str | None = None) -> dict:
        """综合评分：总是返回相似度；有 judge_client 且给了 question 时再返回 Judge 分"""
        out = {"semantic_similarity": self.semantic_similarity(prediction, gold_answer)}
        if self.judge_client is not None and question is not None:
            out["llm_judge"] = self.llm_judge(question, prediction, gold_answer)
        return out

    @staticmethod
    def _cosine(v1: list[float], v2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        return dot / (n1 * n2) if n1 and n2 else 0.0
