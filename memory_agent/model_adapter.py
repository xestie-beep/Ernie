from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib import request

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MODEL_KEEP_ALIVE,
    DEFAULT_OLLAMA_BASE_URL,
    MODEL_REQUEST_TIMEOUT_SECONDS,
)


@dataclass(slots=True)
class ModelMessage:
    role: str
    content: str


@dataclass(slots=True)
class ModelResponse:
    content: str
    model: str | None = None
    role: str = "assistant"
    thinking: str | None = None
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter:
    @property
    def enabled(self) -> bool:
        raise NotImplementedError

    def chat(self, messages: list[ModelMessage]) -> ModelResponse:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        raise NotImplementedError


class DisabledModelAdapter(BaseModelAdapter):
    @property
    def enabled(self) -> bool:
        return False

    def chat(self, messages: list[ModelMessage]) -> ModelResponse:
        raise RuntimeError("No chat model is configured.")

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "backend": None,
            "model": None,
        }


class OllamaChatAdapter(BaseModelAdapter):
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = MODEL_REQUEST_TIMEOUT_SECONDS,
        keep_alive: str | None = None,
        fetch_response: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.model = (model or DEFAULT_CHAT_MODEL).strip()
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = (
            keep_alive.strip()
            if keep_alive is not None
            else DEFAULT_MODEL_KEEP_ALIVE
        )
        self._fetch_response_override = fetch_response

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def chat(self, messages: list[ModelMessage]) -> ModelResponse:
        if not self.enabled:
            raise RuntimeError("No Ollama chat model is configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive

        data = self._fetch_response(payload)
        message = data.get("message", {})
        content = str(message.get("content") or "").strip()
        if not content:
            raise RuntimeError("Model response was empty.")
        return ModelResponse(
            content=content,
            model=str(data.get("model") or self.model),
            role=str(message.get("role") or "assistant"),
            thinking=(
                str(message.get("thinking")).strip()
                if message.get("thinking") is not None
                else None
            ),
            done_reason=(
                str(data.get("done_reason")).strip()
                if data.get("done_reason") is not None
                else None
            ),
            prompt_eval_count=(
                int(data["prompt_eval_count"])
                if isinstance(data.get("prompt_eval_count"), int)
                else None
            ),
            eval_count=(
                int(data["eval_count"])
                if isinstance(data.get("eval_count"), int)
                else None
            ),
            raw=data,
        )

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": "ollama",
            "model": self.model or None,
            "base_url": self.base_url,
            "keep_alive": self.keep_alive or None,
            "timeout_seconds": self.timeout_seconds,
        }

    def _fetch_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._fetch_response_override is not None:
            return self._fetch_response_override(payload)

        req = request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def build_default_model_adapter() -> BaseModelAdapter:
    adapter = OllamaChatAdapter()
    if adapter.enabled:
        return adapter
    return DisabledModelAdapter()
