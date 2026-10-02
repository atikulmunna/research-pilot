"""Difficulty-based model router.

Every call names a task kind. The router looks the task up in the registry, derives the
tier (lite / strong / coding) and reasoning effort from its difficulty, and dispatches to
the provider configured for that tier. Agents never pick models themselves.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..state.models import now_iso
from .json_utils import parse_json_object
from .mock import MockResponder
from .providers import Completion, ProviderClient, normalize_provider
from .schema import render_schema
from .tasks import TASKS, Difficulty, Tier, effort_for, get_task, parse_overrides, resolve_tier

T = TypeVar("T", bound=BaseModel)
EFFORTS = {"low", "medium", "high"}
# Lite-tier model used when the default provider has an obvious cheap sibling and no lite model is set.
LITE_DEFAULTS = {"anthropic": "claude-haiku-4-5"}
# Anthropic first-party prices in USD per million tokens (input, output), cached 2026-09-25.
# The API does not report cost, so budgets rely on this table. Override with LLM_PRICING.
DEFAULT_PRICING: Dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class LLMOutputError(RuntimeError):
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class TierConfig:
    provider: str
    model: str
    effort: str
    temperature: float | None
    max_tokens: int | None


@dataclass(frozen=True)
class Route:
    task: str
    label: str
    difficulty: Difficulty
    tier: Tier
    provider: str
    model: str
    reasoning_effort: str | None
    temperature: float | None = None
    max_tokens: int | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "label": self.label,
            "difficulty": self.difficulty.label,
            "tier": self.tier.value,
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort or "",
        }


@dataclass
class CallRecord:
    task: str
    tier: str
    difficulty: str
    provider: str
    model: str
    reasoning_effort: str
    success: bool
    latency_ms: float
    escalated: bool = False
    fallback: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=now_iso)


def _parse_float(raw: str) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    return float(text)


def parse_pricing(raw: str) -> Dict[str, tuple[float, float]]:
    if not (raw or "").strip():
        return {}
    data = json.loads(raw)
    out: Dict[str, tuple[float, float]] = {}
    for model, value in data.items():
        if isinstance(value, dict):
            out[model] = (float(value.get("input", 0.0)), float(value.get("output", 0.0)))
        else:
            out[model] = (float(value[0]), float(value[1]))
    return out


class ModelRouter:
    def __init__(
        self,
        settings: Settings,
        client: ProviderClient | None = None,
        mock: MockResponder | None = None,
    ):
        self.settings = settings
        self.client = client or ProviderClient(
            api_keys={"anthropic": settings.anthropic_api_key, "openrouter": settings.openrouter_api_key},
            retry_max_attempts=settings.llm_retry_max_attempts,
            retry_base_delay_s=settings.llm_retry_base_delay_s,
            retry_max_delay_s=settings.llm_retry_max_delay_s,
            timeout_s=settings.llm_request_timeout_s,
        )
        self.mock = mock or MockResponder()
        self.strong_min = Difficulty.parse(settings.llm_strong_min_difficulty)
        self.overrides = parse_overrides(settings.llm_task_overrides)
        self.pricing = {**DEFAULT_PRICING, **parse_pricing(settings.llm_pricing)}
        self.listeners: List[Callable[[CallRecord], None]] = []
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    # ------------------------------------------------------------ routing

    def tier_config(self, tier: Tier) -> TierConfig:
        s = self.settings
        default = (normalize_provider(s.llm_provider), s.llm_model.strip())

        def own(prefix: str) -> tuple[str, str] | None:
            model = getattr(s, f"llm_{prefix}_model").strip()
            if not model:
                return None
            provider = normalize_provider(getattr(s, f"llm_{prefix}_provider")) or default[0]
            return provider, model

        if tier is Tier.LITE:
            lite_default = (default[0], LITE_DEFAULTS.get(default[0], default[1]))
            provider, model = own("lite") or lite_default
        elif tier is Tier.STRONG:
            provider, model = own("strong") or default
        elif tier is Tier.CODING:
            provider, model = own("coding") or own("strong") or default
        else:
            return TierConfig("code", "", "none", None, None)
        prefix = tier.value
        max_tokens = int(getattr(s, f"llm_{prefix}_max_tokens") or 0) or None
        return TierConfig(
            provider=provider,
            model=model,
            effort=str(getattr(s, f"llm_{prefix}_reasoning_effort") or "").strip().lower(),
            temperature=_parse_float(getattr(s, f"llm_{prefix}_temperature")),
            max_tokens=max_tokens,
        )

    def resolve(self, task_kind: str, tier: Tier | None = None) -> Route:
        spec = get_task(task_kind)
        chosen = tier or resolve_tier(spec, self.strong_min, self.overrides)
        cfg = self.tier_config(chosen)
        if chosen is Tier.CODE:
            return Route(spec.kind, spec.label, spec.difficulty, chosen, "code", "", None)
        if cfg.effort == "auto":
            effort: str | None = effort_for(spec.difficulty)
        elif cfg.effort in EFFORTS:
            effort = cfg.effort
        else:
            effort = None
        return Route(
            task=spec.kind,
            label=spec.label,
            difficulty=spec.difficulty,
            tier=chosen,
            provider=cfg.provider,
            model=cfg.model,
            reasoning_effort=effort,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

    def routing_table(self) -> List[Dict[str, Any]]:
        return [self.resolve(kind).as_dict() for kind in TASKS]

    # ------------------------------------------------------------ calls

    def text(self, task_kind: str, system: str, user: str, context: Dict[str, Any] | None = None) -> str:
        route = self.resolve(task_kind)
        if route.provider == "mock":
            return str(self._mock(route, context))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._complete(route, messages).text

    def json(
        self,
        task_kind: str,
        system: str,
        user: str,
        schema: Type[T],
        context: Dict[str, Any] | None = None,
    ) -> T:
        route = self.resolve(task_kind)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{user}\n\n{render_schema(schema)}"},
        ]
        try:
            return self._json_attempt(route, messages, schema, context)
        except LLMOutputError as exc:
            retry_route = route
            escalated = False
            if self.settings.llm_escalate_on_failure and route.tier is Tier.LITE:
                retry_route = self.resolve(task_kind, tier=Tier.STRONG)
                escalated = True
            feedback = [
                {"role": "assistant", "content": (exc.raw or "")[:4000]},
                {
                    "role": "user",
                    "content": f"That answer was not usable ({exc}). Return only the corrected JSON object.",
                },
            ]
            return self._json_attempt(retry_route, messages + feedback, schema, context, escalated=escalated)

    def _json_attempt(
        self,
        route: Route,
        messages: List[Dict[str, str]],
        schema: Type[T],
        context: Dict[str, Any] | None,
        escalated: bool = False,
    ) -> T:
        if route.provider == "mock":
            return schema.model_validate(self._mock(route, context))
        completion = self._complete(route, messages, escalated=escalated)
        data = parse_json_object(completion.text)
        if data is None:
            data = parse_json_object(self._repair_json(completion.text))
        if data is None:
            raise LLMOutputError("response was not parseable JSON", raw=completion.text)
        data = _unwrap(data, schema)
        if not set(data) & set(schema.model_fields):
            raise LLMOutputError("JSON did not contain any expected fields", raw=completion.text)
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMOutputError(f"schema validation failed: {str(exc)[:400]}", raw=completion.text) from exc

    def _repair_json(self, text: str) -> str:
        route = self.resolve("json.repair")
        if route.provider == "mock":
            return text
        messages = [
            {"role": "system", "content": "You convert text into one strict JSON object. Output JSON only."},
            {"role": "user", "content": f"Convert this into a single valid JSON object:\n\n{text[:10000]}"},
        ]
        try:
            return self._complete(route, messages).text
        except Exception:
            return ""

    def _mock(self, route: Route, context: Dict[str, Any] | None) -> Any:
        start = time.perf_counter()
        result = self.mock.respond(route.task, context or {})
        self._record(
            CallRecord(
                task=route.task,
                tier=route.tier.value,
                difficulty=route.difficulty.label,
                provider="mock",
                model=route.model,
                reasoning_effort=route.reasoning_effort or "",
                success=True,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        )
        return result

    def _complete(self, route: Route, messages: List[Dict[str, str]], escalated: bool = False) -> Completion:
        candidates = [(route.provider, route.model)]
        fallback_model = self.settings.llm_fallback_model.strip()
        if self.settings.llm_route_fallback_enabled and fallback_model:
            fallback = (normalize_provider(self.settings.llm_fallback_provider) or route.provider, fallback_model)
            if fallback not in candidates:
                candidates.append(fallback)
        for idx, (provider, model) in enumerate(candidates):
            start = time.perf_counter()
            try:
                completion = self.client.complete(
                    provider,
                    model,
                    messages,
                    reasoning_effort=route.reasoning_effort,
                    temperature=route.temperature,
                    max_tokens=route.max_tokens,
                )
            except Exception as exc:
                self._record(
                    CallRecord(
                        task=route.task,
                        tier=route.tier.value,
                        difficulty=route.difficulty.label,
                        provider=provider,
                        model=model,
                        reasoning_effort=route.reasoning_effort or "",
                        success=False,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        escalated=escalated,
                        fallback=idx > 0,
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                )
                if idx == len(candidates) - 1:
                    raise
                continue
            cost = completion.cost_usd or self._estimate_cost(completion.model or model, completion)
            self._record(
                CallRecord(
                    task=route.task,
                    tier=route.tier.value,
                    difficulty=route.difficulty.label,
                    provider=provider,
                    model=model,
                    reasoning_effort=route.reasoning_effort or "",
                    success=True,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    escalated=escalated,
                    fallback=idx > 0,
                    prompt_tokens=completion.prompt_tokens,
                    completion_tokens=completion.completion_tokens,
                    total_tokens=completion.total_tokens,
                    cost_usd=cost,
                )
            )
            return completion
        raise RuntimeError("No completion route available.")

    def _estimate_cost(self, model: str, completion: Completion) -> float:
        price = self.pricing.get(model) or self.pricing.get(model.split("/")[-1])
        if not price:
            return 0.0
        return (completion.prompt_tokens * price[0] + completion.completion_tokens * price[1]) / 1_000_000

    # ------------------------------------------------------------ metrics

    def _record(self, record: CallRecord) -> None:
        key = f"{record.task}|{record.tier}|{record.provider}|{record.model}"
        with self._lock:
            row = self.metrics.setdefault(
                key,
                {
                    "task": record.task,
                    "tier": record.tier,
                    "provider": record.provider,
                    "model": record.model,
                    "calls": 0,
                    "successes": 0,
                    "failures": 0,
                    "escalations": 0,
                    "fallback_calls": 0,
                    "total_latency_ms": 0.0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "last_error": "",
                },
            )
            row["calls"] += 1
            row["successes" if record.success else "failures"] += 1
            row["escalations"] += int(record.escalated)
            row["fallback_calls"] += int(record.fallback)
            row["total_latency_ms"] += record.latency_ms
            row["total_tokens"] += record.total_tokens
            row["cost_usd"] += record.cost_usd
            if record.error:
                row["last_error"] = record.error
        for listener in list(self.listeners):
            try:
                listener(record)
            except Exception:
                continue

    def usage_by_tier(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for row in self.metrics.values():
                tier = out.setdefault(row["tier"], {"calls": 0, "total_tokens": 0, "cost_usd": 0.0, "escalations": 0})
                tier["calls"] += row["calls"]
                tier["total_tokens"] += row["total_tokens"]
                tier["cost_usd"] += row["cost_usd"]
                tier["escalations"] += row["escalations"]
        return out


def _unwrap(data: Dict[str, Any], schema: Type[BaseModel]) -> Dict[str, Any]:
    fields = set(schema.model_fields)
    if set(data) & fields:
        return data
    if len(data) == 1:
        inner = next(iter(data.values()))
        if isinstance(inner, dict) and set(inner) & fields:
            return inner
    return data


def call_record_dict(record: CallRecord) -> Dict[str, Any]:
    return asdict(record)
