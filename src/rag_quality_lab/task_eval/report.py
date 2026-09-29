"""Static reports for recorded complete-task evaluations."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from rag_quality_lab.domain.models import canonical_hash
from rag_quality_lab.task_eval.core import Evaluation


def write_report(result: Evaluation, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "task-" + canonical_hash(result.model_dump(mode="json"))[:16]
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"
    json_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).parent.parent / "reporting" / "templates"),
        autoescape=True,
    )
    html_path.write_text(
        environment.get_template("task_eval.html.jinja2").render(report=result),
        encoding="utf-8",
    )
    return json_path, html_path
