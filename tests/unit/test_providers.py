import inspect
from types import SimpleNamespace

import pytest
import requests
from anthropic.resources.beta.messages import Messages as BetaMessages
from anthropic.resources.messages import Messages as StableMessages

from research_pilot.llm.providers import FALLBACK_BETA, ModelRefusal, ModelTruncated, ProviderClient

MSG = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


# ---------------------------------------------------------------- anthropic


def claude_response(text="ok", stop_reason="end_turn", model="claude-opus-5-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        model=model,
        usage=SimpleNamespace(input_tokens=100, output_tokens=40, cache_creation_input_tokens=0, cache_read_input_tokens=10),
    )


class FakeStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.response


class FakeAnthropic:
    def __init__(self, response=None):
        self.calls = []
        response = response or claude_response()
        self.messages = SimpleNamespace(stream=lambda **kw: self._record("stable", kw, response))
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: self._record("beta", kw, response)))

    def _record(self, surface, params, response):
        # Reject arguments the real SDK method would reject, so signature drift fails here.
        method = BetaMessages.stream if surface == "beta" else StableMessages.stream
        unknown = set(params) - set(inspect.signature(method).parameters)
        if unknown:
            raise TypeError(f"{surface} messages.stream() got unexpected arguments {sorted(unknown)}")
        self.calls.append((surface, params))
        return FakeStream(response)


def test_opus_uses_adaptive_thinking_effort_and_default_fallbacks():
    fake = FakeAnthropic()
    out = ProviderClient(anthropic_client=fake).complete("anthropic", "claude-opus-5-5", MSG, reasoning_effort="high", temperature=0.3)
    surface, params = fake.calls[0]
    assert surface == "beta"
    assert params["betas"] == [FALLBACK_BETA] and params["fallbacks"] == "default"
    assert params["thinking"] == {"type": "adaptive"} and params["output_config"] == {"effort": "high"}
    assert params["system"] == "be brief" and params["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in params and params["max_tokens"] == 64000
    assert out.text == "ok" and out.prompt_tokens == 110 and out.completion_tokens == 40 and out.total_tokens == 150


def test_haiku_uses_thinking_budget_only_above_low_effort():
    fake = FakeAnthropic(claude_response(model="claude-haiku-4-5"))
    client = ProviderClient(anthropic_client=fake)
    client.complete("anthropic", "claude-haiku-4-5", MSG, reasoning_effort="low", temperature=0.1)
    client.complete("anthropic", "claude-haiku-4-5", MSG, reasoning_effort="high", max_tokens=4000)
    (s1, low), (s2, high) = fake.calls
    assert s1 == s2 == "stable"
    assert "thinking" not in low and "output_config" not in low and "temperature" not in low
    assert high["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert high["max_tokens"] > 8192


def test_models_without_fallback_support_use_the_stable_endpoint():
    fake = FakeAnthropic()
    ProviderClient(anthropic_client=fake).complete("claude", "claude-sonnet-4-6", MSG, reasoning_effort="medium")
    surface, params = fake.calls[0]
    assert surface == "stable" and "fallbacks" not in params
    assert params["output_config"] == {"effort": "medium"}


def test_truncated_answer_raises():
    fake = FakeAnthropic(claude_response(text='{"partial": ', stop_reason="max_tokens"))
    with pytest.raises(ModelTruncated, match="max_tokens=64000"):
        ProviderClient(anthropic_client=fake).complete("anthropic", "claude-opus-5-5", MSG)


def test_refusal_raises():
    fake = FakeAnthropic(claude_response(text="", stop_reason="refusal"))
    with pytest.raises(ModelRefusal, match="cyber"):
        ProviderClient(anthropic_client=fake).complete("anthropic", "claude-opus-5-5", MSG)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="anthropic, openrouter or mock"):
        ProviderClient().complete("groq", "m", MSG)


# ---------------------------------------------------------------- openrouter


class FakeResponse:
    def __init__(self, data=None, status=200, headers=None):
        self._data = data or {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost": 0.002},
        }
        self.status_code = status
        self.headers = headers or {}
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return self._data


def openrouter(**kw):
    return ProviderClient(api_keys={"openrouter": "k"}, retry_base_delay_s=0, retry_max_delay_s=0, **kw)


def test_openrouter_payload_and_cost(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=0):
        calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse()

    monkeypatch.setattr("research_pilot.llm.providers.requests.post", fake_post)
    out = openrouter().complete("openrouter", "anthropic/claude-haiku-4.5", MSG, reasoning_effort="low", temperature=0.1, max_tokens=50)
    payload = calls[0]["json"]
    assert calls[0]["headers"]["Authorization"] == "Bearer k"
    assert payload["reasoning"] == {"effort": "low"} and payload["usage"] == {"include": True}
    assert payload["temperature"] == 0.1 and payload["max_tokens"] == 50
    assert out.total_tokens == 5 and out.cost_usd == 0.002


def test_openrouter_defaults_max_tokens_and_detects_truncation(monkeypatch):
    sent = []
    truncated = {"choices": [{"message": {"content": '{"partial": '}, "finish_reason": "length"}], "usage": {}}

    def fake_post(url, headers=None, json=None, timeout=0):
        sent.append(json)
        return FakeResponse(truncated)

    monkeypatch.setattr("research_pilot.llm.providers.requests.post", fake_post)
    with pytest.raises(ModelTruncated, match="max_tokens=64000"):
        openrouter().complete("openrouter", "openai/gpt-6-luna", MSG)
    assert sent[0]["max_tokens"] == 64000 and "temperature" not in sent[0]


def test_openrouter_requires_key():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        ProviderClient().complete("openrouter", "m", MSG)


def test_openrouter_retries_429_and_503(monkeypatch):
    responses = [FakeResponse(status=429, headers={"Retry-After": "0"}), FakeResponse(status=503), FakeResponse()]
    sleeps = []
    monkeypatch.setattr("research_pilot.llm.providers.requests.post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr("research_pilot.llm.providers.time.sleep", sleeps.append)
    assert openrouter().complete("or", "m", MSG).text == "ok"
    assert len(sleeps) == 2


def test_openrouter_does_not_retry_400(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(status=400)

    monkeypatch.setattr("research_pilot.llm.providers.requests.post", fake_post)
    with pytest.raises(requests.HTTPError):
        openrouter().complete("openrouter", "m", MSG)
    assert len(calls) == 1
