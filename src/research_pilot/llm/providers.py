"""Chat-completion providers: Anthropic (official SDK) and OpenRouter (HTTP)."""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import anthropic
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER_ALIASES = {"or": "openrouter", "claude": "anthropic"}
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Current Claude models take adaptive thinking plus output_config.effort. Haiku 4.5 and
# pre-4.6 models instead take an explicit thinking budget and no effort parameter.
LEGACY_THINKING_MARKERS = ("haiku", "-3-", "-4-0", "-4-1", "-4-5")
LEGACY_THINKING_BUDGET = {"medium": 2048, "high": 8192}
# Models that accept server-side refusal fallbacks in the "default" form.
FALLBACK_MODELS = ("claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Streaming lets long, thinking-heavy answers finish without HTTP timeouts.
DEFAULT_MAX_TOKENS = 64000


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    raw: Any = field(default=None, repr=False)


class ModelRefusal(RuntimeError):
    pass


class ModelTruncated(RuntimeError):
    pass


def normalize_provider(name: str) -> str:
    key = (name or "").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def uses_thinking_budget(model: str) -> bool:
    lowered = model.lower()
    return any(marker in lowered for marker in LEGACY_THINKING_MARKERS)


class ProviderClient:
    def __init__(
        self,
        api_keys: Dict[str, str] | None = None,
        retry_max_attempts: int = 4,
        retry_base_delay_s: float = 1.0,
        retry_max_delay_s: float = 8.0,
        timeout_s: float = 180.0,
        anthropic_client: Any = None,
    ):
        self.api_keys = {normalize_provider(k): v for k, v in (api_keys or {}).items()}
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        self.retry_max_delay_s = max(0.0, float(retry_max_delay_s))
        self.timeout_s = timeout_s
        self._anthropic = anthropic_client

    def complete(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        name = normalize_provider(provider)
        if name == "anthropic":
            return self._anthropic_complete(model, messages, reasoning_effort, temperature, max_tokens)
        if name == "openrouter":
            return self._openrouter_complete(model, messages, reasoning_effort, temperature, max_tokens)
        raise ValueError(f"Unsupported LLM provider: {provider} (use anthropic, openrouter or mock)")

    # ------------------------------------------------------------ anthropic

    def anthropic_client(self) -> Any:
        if self._anthropic is None:
            key = self.api_keys.get("anthropic", "")
            options = {"max_retries": self.retry_max_attempts - 1, "timeout": self.timeout_s}
            # Without an explicit key the SDK resolves ANTHROPIC_API_KEY or an `ant auth login` profile.
            self._anthropic = anthropic.Anthropic(api_key=key, **options) if key else anthropic.Anthropic(**options)
        return self._anthropic

    def _anthropic_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        effort: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        params: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            params["system"] = system
        client = self.anthropic_client()
        # The 1.x SDK no longer accepts sampling parameters, so temperature is not sent to Anthropic.
        open_stream, extra = client.messages.stream, {}
        if uses_thinking_budget(model):
            budget = LEGACY_THINKING_BUDGET.get(effort or "")
            if budget:
                params["thinking"] = {"type": "enabled", "budget_tokens": budget}
                params["max_tokens"] = max(params["max_tokens"], budget + 4096)
        else:
            params["thinking"] = {"type": "adaptive"}
            if effort:
                params["output_config"] = {"effort": effort}
            if model in FALLBACK_MODELS:
                open_stream, extra = client.beta.messages.stream, {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        with open_stream(**params, **extra) as stream:
            response = stream.get_final_message()
        if response.stop_reason == "max_tokens":
            raise ModelTruncated(f"{model} stopped at max_tokens={params['max_tokens']} before finishing its answer")
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ModelRefusal(f"{model} declined the request (category: {category or 'unspecified'})")
        usage = response.usage
        prompt_tokens = (
            int(usage.input_tokens or 0)
            + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            + int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        completion_tokens = int(usage.output_tokens or 0)
        return Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            provider="anthropic",
            model=str(getattr(response, "model", "") or model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw=response,
        )

    # ------------------------------------------------------------ openrouter

    def _openrouter_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        effort: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> Completion:
        token = self.api_keys.get("openrouter", "")
        if not token:
            raise ValueError("OPENROUTER_API_KEY is required for openrouter provider.")
        payload: Dict[str, Any] = {"model": model, "messages": messages, "usage": {"include": True}}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if effort:
            payload["reasoning"] = {"effort": effort}
        data = self._post_with_retry(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            payload=payload,
        )
        choices = data.get("choices") or [{}]
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = data.get("usage") or {}
        prompt_tokens = int(float(usage.get("prompt_tokens", 0) or 0))
        completion_tokens = int(float(usage.get("completion_tokens", 0) or 0))
        return Completion(
            text=str(content or ""),
            provider="openrouter",
            model=str(data.get("model") or model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=int(float(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)),
            cost_usd=float(usage.get("total_cost", usage.get("cost", 0.0)) or 0.0),
            raw=data,
        )

    def _post_with_retry(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.retry_max_attempts):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if not self._should_retry(exc, status) or attempt >= self.retry_max_attempts - 1:
                    raise
                time.sleep(self._retry_delay_seconds(exc, attempt))
        raise last_exc or RuntimeError("Request failed without exception details.")

    def _should_retry(self, exc: requests.RequestException, status: int | None) -> bool:
        if status is None:
            return isinstance(exc, (requests.Timeout, requests.ConnectionError))
        return status in RETRY_STATUSES

    def _retry_delay_seconds(self, exc: requests.RequestException, attempt: int) -> float:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
                if retry_after >= 0:
                    return min(self.retry_max_delay_s, retry_after)
            except (TypeError, ValueError):
                pass
        exp = self.retry_base_delay_s * (2**attempt)
        jitter = random.uniform(0, min(0.5, self.retry_base_delay_s + 0.1))
        return min(self.retry_max_delay_s, exp + jitter)
