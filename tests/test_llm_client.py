"""模型调用层单元测试

全程 Mock HTTP 层，不消耗真实 API 额度。
覆盖：正常调用 / 429 重试 / 重试耗尽 / 4xx 不重试 / 缺 Key 报错 / 切换模型。
"""
from unittest.mock import Mock, patch

import pytest
import requests

from src.client.llm_client import LLMClient


def _ok_response(content: str = "你好") -> Mock:
    """构造一个 200 的假响应"""
    mock_resp = Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp.raise_for_status = Mock()
    return mock_resp


def _rate_limited() -> Mock:
    mock_resp = Mock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limit"
    mock_resp.raise_for_status = Mock()
    return mock_resp


@pytest.fixture
def client() -> LLMClient:
    return LLMClient(api_key="sk-test", base_url="https://fake.api/v1", model="fake-model")


class TestChat:
    def test_chat_success(self, client):
        """正常调用返回 content / usage / latency，且请求体正确"""
        with patch.object(client.session, "post", return_value=_ok_response("答案A")) as mock_post:
            resp = client.chat([{"role": "user", "content": "1+1=?"}])

        assert resp.content == "答案A"
        assert resp.usage["total_tokens"] == 15
        assert resp.latency >= 0

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "fake-model"
        assert payload["messages"][0]["content"] == "1+1=?"

    def test_chat_with_generate_params(self, client):
        """temperature 等生成参数透传到请求体"""
        with patch.object(client.session, "post", return_value=_ok_response()) as mock_post:
            client.chat([{"role": "user", "content": "hi"}], temperature=0.3, max_tokens=64)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 64


class TestRetry:
    def test_retry_on_429(self, client):
        """429 限流 → 重试后成功"""
        with patch.object(client.session, "post",
                          side_effect=[_rate_limited(), _ok_response()]) as mock_post, \
             patch("time.sleep"):
            resp = client.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "你好"
        assert mock_post.call_count == 2

    def test_retry_exhausted_raises(self, client):
        """持续 429 超过 max_retries → RuntimeError"""
        with patch.object(client.session, "post", return_value=_rate_limited()), \
             patch("time.sleep"):
            with pytest.raises(RuntimeError, match="重试"):
                client.chat([{"role": "user", "content": "hi"}])

    def test_4xx_no_retry(self, client):
        """400 参数错误不重试，直接抛 HTTPError"""
        bad = Mock()
        bad.status_code = 400
        bad.text = "bad request"
        bad.raise_for_status = Mock(side_effect=requests.HTTPError("400", response=bad))
        with patch.object(client.session, "post", return_value=bad) as mock_post:
            with pytest.raises(requests.HTTPError):
                client.chat([{"role": "user", "content": "hi"}])
        assert mock_post.call_count == 1


class TestConfig:
    def test_missing_api_key(self):
        """没配置 API Key → 明确的 ValueError"""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API Key"):
                LLMClient(base_url="https://fake.api")

    def test_set_model(self, client):
        client.set_model("other-model")
        assert client.model == "other-model"

    def test_env_fallback(self):
        """不传参时从环境变量读取配置"""
        env = {
            "OPENAI_API_KEY": "sk-env",
            "OPENAI_BASE_URL": "https://env.api/v1/",
            "DEFAULT_MODEL": "env-model",
        }
        with patch.dict("os.environ", env, clear=True):
            c = LLMClient()
        assert c.api_key == "sk-env"
        assert c.base_url == "https://env.api/v1"  # 尾部斜杠被去掉
        assert c.model == "env-model"
