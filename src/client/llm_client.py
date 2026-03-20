"""统一模型调用层

封装 OpenAI 兼容的 /chat/completions 接口，支持：
- 多模型切换（DeepSeek / Qwen / GPT 等，改 base_url + model 即可）
- 指数退避重试（429 限流 / 5xx 服务端错误 / 网络抖动）
- 统一的响应封装（content + usage + latency）
"""
import os
import time
from dataclasses import dataclass, field

import requests


@dataclass
class LLMResponse:
    """一次模型调用的响应封装"""

    content: str
    usage: dict = field(default_factory=dict)  # prompt_tokens / completion_tokens / total_tokens
    latency: float = 0.0                        # 秒
    raw: dict | None = None                     # 原始响应，调试用


class LLMClient:
    """OpenAI 兼容 API 客户端（硅基流动 / DeepSeek / OpenAI 通用）"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("未配置 API Key：请设置 OPENAI_API_KEY（环境变量或 .env 文件）")
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        ).rstrip("/")
        self.model = model or os.getenv("DEFAULT_MODEL", "deepseek-ai/DeepSeek-V3")
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, messages: list[dict], model: str | None = None, **kwargs) -> LLMResponse:
        """发送对话请求，带指数退避重试

        Args:
            messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
            model: 临时切换模型，不传则用初始化时的默认模型
            **kwargs: temperature / max_tokens 等生成参数
        """
        payload = {"model": model or self.model, "messages": messages, **kwargs}
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            start = time.time()
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                # 429 限流 / 5xx 服务端错误：值得重试
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp
                    )
                # 其他 4xx（参数错误、鉴权失败等）：raise_for_status 直接抛，不重试
                resp.raise_for_status()

                data = resp.json()
                return LLMResponse(
                    content=data["choices"][0]["message"]["content"],
                    usage=data.get("usage", {}),
                    latency=time.time() - start,
                    raw=data,
                )
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                # 4xx（除 429）是请求本身的问题，重试无意义
                if status and 400 <= status < 500 and status != 429:
                    raise
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s → 2s → 4s 指数退避

        raise RuntimeError(f"模型调用失败（已重试 {self.max_retries} 次）: {last_err}")

    def set_model(self, model: str) -> None:
        """切换默认模型（A/B 对比测试用）"""
        self.model = model
