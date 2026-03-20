"""Prompt 策略引擎

用 Jinja2 模板渲染三种策略：
- zero_shot：直接问，不给示例
- few_shot：给若干示例，让模型照着回答
- cot (Chain-of-Thought)：要求模型一步步推理再给结论

支持 register_strategy 注册自定义策略，便于扩展。
"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template

# 模板目录：本文件同级的 templates/
TEMPLATE_DIR = Path(__file__).parent / "templates"


class PromptEngine:
    """Prompt 策略渲染器"""

    STRATEGIES = ("zero_shot", "few_shot", "cot")

    def __init__(self, template_dir: str | Path | None = None):
        loader = FileSystemLoader(str(template_dir or TEMPLATE_DIR))
        # keep_trailing_newline 保留模板末尾换行，让输出更可控
        self.env = Environment(loader=loader, keep_trailing_newline=True)
        self._templates = {name: self.env.get_template(f"{name}.jinja2") for name in self.STRATEGIES}

    def render(self, strategy: str, **kwargs) -> str:
        """渲染指定策略的 Prompt

        Args:
            strategy: 策略名（zero_shot / few_shot / cot 或已注册的自定义策略）
            **kwargs: 模板变量，如 question / examples / task_description
        """
        if strategy not in self._templates:
            raise ValueError(f"未知策略: {strategy}，可选: {self.list_strategies()}")
        return self._templates[strategy].render(**kwargs)

    def register_strategy(self, name: str, template_str: str) -> None:
        """注册自定义策略（传入模板字符串，无需建文件）"""
        self._templates[name] = Template(template_str)

    def list_strategies(self) -> list[str]:
        return list(self._templates)
