"""OpenAI-compatible HTTP executor with explicit, non-secret configuration."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .agent_contract import AgentResponse, AgentSpec, AgentTask, Usage


@dataclass(frozen=True)
class TokenPricing:
    """User-supplied USD prices per million tokens; never inferred by Merit."""

    input_per_million: float
    output_per_million: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_per_million) or self.input_per_million < 0:
            raise ValueError("Input token price must be finite and non-negative")
        if not math.isfinite(self.output_per_million) or self.output_per_million < 0:
            raise ValueError("Output token price must be finite and non-negative")


class OpenAICompatibleExecutor:
    """Call a chat-completions endpoint without storing credentials in evidence."""

    name: str
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    _CONFIG_KEYS = ("temperature", "top_p", "max_tokens", "stop", "seed", "response_format")

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        pricing: TokenPricing | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url and model must not be empty")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds must be finite and positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_backoff_seconds < 0 or not math.isfinite(retry_backoff_seconds):
            raise ValueError("retry_backoff_seconds must be finite and non-negative")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai_compatible:{model}"
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.pricing = pricing

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _payload(self, agent: AgentSpec, task: AgentTask) -> dict[str, Any]:
        configured_model = agent.config.get("model", self.model)
        if configured_model != self.model:
            raise ValueError("Executor model is fixed; create a new executor for a different model")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": agent.instructions},
                {"role": "user", "content": task.prompt},
            ],
        }
        for key in self._CONFIG_KEYS:
            if key in agent.config:
                payload[key] = agent.config[key]
        return payload

    def _request(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Merit-Agent-Eval/0.1",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                last_error = error
                status = error.code
                error.close()
                retryable = status in self._RETRYABLE_STATUS
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"Agent HTTP request failed with status {status}") from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    raise RuntimeError("Agent HTTP request failed or returned invalid JSON") from error
            if self.retry_backoff_seconds:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("Agent HTTP request failed") from last_error

    @staticmethod
    def _content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Agent response is missing choices[0].message.content") from error
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            if texts:
                return "".join(texts)
        raise RuntimeError("Agent response content must be a string or text parts")

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in environment variable {self.api_key_env}")
        started = time.perf_counter()
        payload = self._request(self._payload(agent, task), api_key)
        output = self._content(payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_usage = payload.get("usage")
        tokens_reported = isinstance(raw_usage, dict) and all(
            key in raw_usage for key in ("prompt_tokens", "completion_tokens")
        )
        if tokens_reported:
            input_tokens = int(raw_usage["prompt_tokens"])
            output_tokens = int(raw_usage["completion_tokens"])
        else:
            input_tokens = 0
            output_tokens = 0
        cost_reported = tokens_reported and self.pricing is not None
        cost_usd = 0.0
        if cost_reported:
            cost_usd = (
                input_tokens * self.pricing.input_per_million
                + output_tokens * self.pricing.output_per_million
            ) / 1_000_000.0
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            tokens_reported=tokens_reported,
            cost_reported=cost_reported,
            cost_is_estimate=cost_reported,
        )
        return AgentResponse(output=output, usage=usage)
