"""评测执行器

跑数据集 × 策略矩阵，并发调用模型，收集响应，可选用 Scorer 打分。
输出结构化的评测结果，供 Reporter 生成报告。
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_dataset(path: str | Path) -> dict:
    """加载 JSON 格式的评测数据集"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Evaluator:
    """评测执行器：client（模型调用）+ prompt_engine（策略渲染）+ scorer（评分，可选）"""

    def __init__(self, client, prompt_engine, scorer=None):
        self.client = client
        self.prompt_engine = prompt_engine
        self.scorer = scorer

    def _run_one(self, sample, strategy, examples, task_description, gen_params):
        """跑单条样本的单一策略：渲染 Prompt → 调模型 → (可选)打分"""
        prompt = self.prompt_engine.render(
            strategy,
            task_description=task_description,
            question=sample["question"],
            examples=examples,
        )
        resp = self.client.chat([{"role": "user", "content": prompt}], **gen_params)

        result = {
            "id": sample["id"],
            "category": sample.get("category", ""),
            "question": sample["question"],
            "gold_answer": sample["gold_answer"],
            "strategy": strategy,
            "prompt": prompt,
            "response": resp.content,
            "latency": round(resp.latency, 2),
            "usage": resp.usage,
        }
        if self.scorer is not None:
            result["scores"] = self.scorer.score(
                prediction=resp.content,
                gold_answer=sample["gold_answer"],
                question=sample["question"],
            )
        return result

    def evaluate(
        self,
        dataset: dict,
        strategies=("zero_shot", "few_shot", "cot"),
        gen_params: dict | None = None,
        max_workers: int = 4,
    ) -> dict:
        """对整个数据集跑多策略评测

        Args:
            dataset: load_dataset 加载的数据集
            strategies: 要对比的策略列表
            gen_params: 传给模型 chat 的生成参数（temperature 等）
            max_workers: 并发线程数
        """
        gen_params = gen_params or {"temperature": 0.3}
        examples = dataset.get("few_shot_examples", [])
        task_description = dataset.get("task_description", "")
        samples = dataset["samples"]

        # 策略 × 样本 的全组合任务
        tasks = [(s, st) for st in strategies for s in samples]
        results, failures = [], []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._run_one, s, st, examples, task_description, gen_params): (s, st)
                for s, st in tasks
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    s, st = futures[fut]
                    failures.append({"id": s["id"], "strategy": st, "error": str(e)})

        # 按 (策略, id) 排序，便于阅读
        results.sort(key=lambda r: (r["strategy"], r["id"]))

        return {
            "meta": {
                "model": self.client.model,
                "strategies": list(strategies),
                "sample_count": len(samples),
                "total_calls": len(tasks),
                "fail_count": len(failures),
            },
            "results": results,
            "failures": failures,
        }
