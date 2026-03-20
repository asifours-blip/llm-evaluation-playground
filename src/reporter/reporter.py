"""HTML 评测报告生成器

把 Evaluator 的输出渲染成一份可读的 HTML 报告：
- 策略对比表（平均相似度 / Judge 分 / 延迟）+ CSS 柱状可视化
- 每条样本的详细问答
- 失败案例清单
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from jinja2 import Template

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>LLM 评测报告</title>
<style>
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 40px; color: #333; max-width: 1200px; }
  h1 { color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }
  h2 { margin-top: 32px; }
  .meta { color: #666; font-size: 14px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }
  th { background: #f5f5f5; }
  .bar-wrap { background: #eee; border-radius: 3px; width: 100%; }
  .bar { background: #1a73e8; height: 18px; border-radius: 3px; }
  .fail { color: #d93025; }
  td.detail { max-width: 300px; word-break: break-word; }
</style>
</head>
<body>
<h1>LLM 评测报告</h1>
<p class="meta">
  模型：<b>{{ meta.model }}</b> ｜ 样本数：{{ meta.sample_count }} ｜
  总调用：{{ meta.total_calls }} ｜ 失败：{{ meta.fail_count }} ｜
  生成时间：{{ generated_at }}
</p>

<h2>策略对比</h2>
<table>
<tr>
  <th>策略</th><th>样本数</th><th>平均语义相似度</th><th>相似度可视化</th>
  <th>平均 Judge 分</th><th>平均延迟 (s)</th>
</tr>
{% for s in summary %}
<tr>
  <td>{{ s.strategy }}</td>
  <td>{{ s.count }}</td>
  <td>{{ s.avg_similarity if s.avg_similarity is not none else '—' }}</td>
  <td>
    {% if s.avg_similarity is not none %}
    <div class="bar-wrap"><div class="bar" style="width: {{ (s.avg_similarity * 100)|round(1) }}%"></div></div>
    {% endif %}
  </td>
  <td>{{ s.avg_judge if s.avg_judge is not none else '—' }}</td>
  <td>{{ s.avg_latency if s.avg_latency is not none else '—' }}</td>
</tr>
{% endfor %}
</table>

<h2>详细结果</h2>
<table>
<tr>
  <th>ID</th><th>策略</th><th>类别</th><th>问题</th><th>参考答案</th>
  <th>模型回答</th><th>相似度</th><th>Judge</th><th>延迟(s)</th>
</tr>
{% for r in results %}
<tr>
  <td>{{ r.id }}</td>
  <td>{{ r.strategy }}</td>
  <td>{{ r.category }}</td>
  <td class="detail">{{ r.question }}</td>
  <td class="detail">{{ r.gold_answer }}</td>
  <td class="detail">{{ r.response[:200] }}{% if r.response|length > 200 %}…{% endif %}</td>
  <td>{{ r.scores.get('semantic_similarity', '—') if r.scores else '—' }}</td>
  <td>{{ r.scores.get('llm_judge', '—') if r.scores else '—' }}</td>
  <td>{{ r.latency }}</td>
</tr>
{% endfor %}
</table>

{% if failures %}
<h2 class="fail">失败案例 ({{ failures|length }})</h2>
<table>
<tr><th>ID</th><th>策略</th><th>错误信息</th></tr>
{% for f in failures %}
<tr class="fail"><td>{{ f.id }}</td><td>{{ f.strategy }}</td><td>{{ f.error }}</td></tr>
{% endfor %}
</table>
{% endif %}
</body>
</html>""")


class Reporter:
    def generate(self, eval_output: dict, output_path: str | Path = "data/reports/report.html") -> str:
        """生成 HTML 报告并写盘，返回文件路径"""
        summary = self._summarize(eval_output["results"])
        html = HTML_TEMPLATE.render(
            meta=eval_output["meta"],
            summary=summary,
            results=eval_output["results"],
            failures=eval_output.get("failures", []),
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        return str(path)

    def _summarize(self, results: list[dict]) -> list[dict]:
        """按策略聚合：平均相似度 / Judge / 延迟"""
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_strategy[r["strategy"]].append(r)

        summary = []
        for strategy, items in by_strategy.items():
            sims = [r["scores"]["semantic_similarity"] for r in items
                    if r.get("scores", {}).get("semantic_similarity") is not None]
            judges = [r["scores"]["llm_judge"] for r in items
                      if r.get("scores", {}).get("llm_judge") is not None]
            lats = [r["latency"] for r in items]
            summary.append({
                "strategy": strategy,
                "count": len(items),
                "avg_similarity": round(sum(sims) / len(sims), 4) if sims else None,
                "avg_judge": round(sum(judges) / len(judges), 2) if judges else None,
                "avg_latency": round(sum(lats) / len(lats), 2) if lats else None,
            })
        return summary
