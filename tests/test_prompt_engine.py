"""Prompt 引擎单元测试"""
import pytest

from src.prompt.prompt_engine import PromptEngine


@pytest.fixture
def engine():
    return PromptEngine()


class TestRender:
    def test_zero_shot_contains_question(self, engine):
        """zero_shot 渲染结果含问题、不含示例"""
        out = engine.render("zero_shot", question="1+1=?", task_description="答题")
        assert "1+1=?" in out
        assert "示例" not in out

    def test_few_shot_includes_examples(self, engine):
        """few_shot 渲染结果含示例的问答对"""
        examples = [
            {"question": "A?", "answer": "a"},
            {"question": "B?", "answer": "b"},
        ]
        out = engine.render("few_shot", question="C?", examples=examples, task_description="答题")
        assert "A?" in out and "a" in out
        assert "B?" in out and "b" in out
        assert "C?" in out

    def test_cot_contains_reasoning_hint(self, engine):
        """cot 模板含逐步推理的提示"""
        out = engine.render("cot", question="15×12=?", task_description="答题")
        assert "15×12=?" in out
        assert "一步步" in out or "推理" in out or "答案：" in out

    def test_task_description_default(self, engine):
        """不传 task_description 时用默认值，不报错"""
        out = engine.render("zero_shot", question="q")
        assert "q" in out


class TestExtensibility:
    def test_unknown_strategy_raises(self, engine):
        with pytest.raises(ValueError, match="未知策略"):
            engine.render("not_a_strategy", question="q")

    def test_register_custom_strategy(self, engine):
        engine.register_strategy("upper", "{{ question | upper }}")
        out = engine.render("upper", question="abc")
        assert out == "ABC"

    def test_list_strategies(self, engine):
        names = engine.list_strategies()
        assert set(("zero_shot", "few_shot", "cot")).issubset(set(names))
