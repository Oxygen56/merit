"""Local Ollama executor with resumable response caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .agent_contract import AgentResponse, AgentSpec, AgentTask, Usage
from .openai_executor import OpenAICompatibleExecutor


class OllamaExecutor(OpenAICompatibleExecutor):
    """Call only a loopback Ollama endpoint; no API key is read or sent."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434/v1",
        cache_dir: Path | None = None,
        max_network_calls: int | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
    ) -> None:
        hostname = urlparse(base_url).hostname
        if hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("OllamaExecutor only accepts a loopback base_url")
        if max_network_calls is not None and max_network_calls < 1:
            raise ValueError("max_network_calls must be positive")
        super().__init__(
            base_url=base_url,
            model=model,
            api_key_env="",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.name = f"ollama:{model}"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_network_calls = max_network_calls
        self.network_calls = 0
        self.cache_hits = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, payload: dict[str, Any]) -> Path | None:
        if not self.cache_dir:
            return None
        identity = json.dumps(
            {"endpoint": self.endpoint, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return self.cache_dir / f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.json"

    @staticmethod
    def _to_response(payload: dict[str, Any], latency_ms: float) -> AgentResponse:
        output = OpenAICompatibleExecutor._content(payload)
        raw_usage = payload.get("usage")
        reported = isinstance(raw_usage, dict) and all(
            key in raw_usage for key in ("prompt_tokens", "completion_tokens")
        )
        return AgentResponse(
            output=output,
            usage=Usage(
                input_tokens=int(raw_usage["prompt_tokens"]) if reported else 0,
                output_tokens=int(raw_usage["completion_tokens"]) if reported else 0,
                latency_ms=latency_ms,
                tokens_reported=reported,
                cost_reported=False,
            ),
        )

    def invoke(self, agent: AgentSpec, task: AgentTask) -> AgentResponse:
        payload = self._payload(agent, task)
        cache_path = self._cache_path(payload)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache_hits += 1
            return self._to_response(cached["response"], float(cached["latency_ms"]))
        if self.max_network_calls is not None and self.network_calls >= self.max_network_calls:
            raise RuntimeError(f"Ollama network call limit reached: {self.max_network_calls}")
        started = time.perf_counter()
        response = self._request(payload, "")
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.network_calls += 1
        if cache_path:
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"response": response, "latency_ms": latency_ms}, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        return self._to_response(response, latency_ms)
