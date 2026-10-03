from research_pilot.config import Settings
from research_pilot.llm.json_utils import extract_code_block, parse_json_object
from research_pilot.llm.providers import Completion
from research_pilot.llm.router import ModelRouter
from research_pilot.state.models import Gap, GapAnalysis, QueryPlan


class ScriptedClient:
    """Returns queued texts and records which provider/model handled each call."""

    def __init__(self, texts, fail_models=()):
        self.texts = list(texts)
        self.fail_models = set(fail_models)
        self.calls = []

    def complete(self, provider, model, messages, reasoning_effort=None, temperature=None, max_tokens=None):
        self.calls.append({"provider": provider, "model": model, "effort": reasoning_effort, "messages": messages})
        if model in self.fail_models:
            raise RuntimeError(f"{model} down")
        return Completion(text=self.texts.pop(0), provider=provider, model=model, prompt_tokens=10, completion_tokens=5, total_tokens=15)


def settings(**overrides):
    values = dict(
        _env_file=None,
        llm_provider="openrouter",
        llm_model="default",
        llm_lite_model="lite-m",
        llm_standard_model="standard-m",
        llm_strong_model="strong-m",
    )
    values.update(overrides)
    return Settings(**values)


def test_json_is_parsed_from_fenced_output_and_validated():
    client = ScriptedClient(['Here you go:\n```json\n{"queries": ["a", "b"], "synonyms": "c"}\n```'])
    out = ModelRouter(settings(), client=client).json("field.search_queries", "sys", "user", QueryPlan)
    assert out.queries == ["a", "b"]
    assert out.synonyms == ["c"]
    assert client.calls[0]["model"] == "lite-m"
    assert client.calls[0]["effort"] == "low"
    assert '"queries"' in client.calls[0]["messages"][1]["content"]


def test_unparseable_output_is_repaired_on_the_lite_tier():
    client = ScriptedClient(["queries are a and b", '{"queries": ["a", "b"]}'])
    out = ModelRouter(settings(), client=client).json("field.search_queries", "s", "u", QueryPlan)
    assert out.queries == ["a", "b"]
    assert [c["model"] for c in client.calls] == ["lite-m", "lite-m"]


def test_lite_failure_escalates_to_standard():
    client = ScriptedClient(["nothing useful", "still nothing", '{"queries": ["x"]}'])
    router = ModelRouter(settings(), client=client)
    out = router.json("field.search_queries", "s", "u", QueryPlan)
    assert out.queries == ["x"]
    assert client.calls[-1]["model"] == "standard-m"
    assert router.usage_by_tier()["standard"]["escalations"] == 1


def test_standard_failure_escalates_to_strong():
    client = ScriptedClient(["no json here", "no json either", '{"gaps": []}'])
    router = ModelRouter(settings(), client=client)
    router.json("gap.discovery", "s", "u", GapAnalysis)
    assert [c["model"] for c in client.calls] == ["standard-m", "lite-m", "strong-m"]
    assert router.usage_by_tier()["strong"]["escalations"] == 1


def test_escalation_can_be_disabled():
    client = ScriptedClient(["nope", "nope", '{"queries": ["x"]}'])
    ModelRouter(settings(llm_escalate_on_failure=False), client=client).json("field.search_queries", "s", "u", QueryPlan)
    assert {c["model"] for c in client.calls} == {"lite-m"}


def test_high_and_very_high_tasks_use_standard_and_strong_models():
    client = ScriptedClient(['{"gaps": [{"description": "d", "novelty_strength": "high"}]}', '{"gaps": []}'])
    router = ModelRouter(settings(), client=client)
    out = router.json("gap.discovery", "s", "u", GapAnalysis)
    router.json("novelty.analysis", "s", "u", GapAnalysis)
    assert out.gaps[0].novelty_strength == "strong"
    assert [(c["model"], c["effort"]) for c in client.calls] == [("standard-m", "high"), ("strong-m", "high")]


def test_fallback_model_used_when_primary_fails():
    client = ScriptedClient(['{"queries": ["q"]}'], fail_models={"lite-m"})
    router = ModelRouter(settings(llm_fallback_provider="anthropic", llm_fallback_model="backup"), client=client)
    assert router.json("field.search_queries", "s", "u", QueryPlan).queries == ["q"]
    assert [(c["provider"], c["model"]) for c in client.calls] == [("openrouter", "lite-m"), ("anthropic", "backup")]


def test_listeners_receive_call_records_with_cost_estimates():
    records = []
    client = ScriptedClient(['{"queries": ["q"]}'])
    router = ModelRouter(settings(llm_pricing='{"lite-m": [1.0, 2.0]}'), client=client)
    router.listeners.append(records.append)
    router.json("field.search_queries", "s", "u", QueryPlan)
    assert records[0].tier == "lite"
    assert records[0].total_tokens == 15
    assert abs(records[0].cost_usd - (10 * 1.0 + 5 * 2.0) / 1_000_000) < 1e-12


def test_claude_cost_comes_from_builtin_pricing():
    records = []

    class ClaudeClient(ScriptedClient):
        def complete(self, provider, model, messages, **kw):
            out = super().complete(provider, model, messages, **kw)
            out.prompt_tokens, out.completion_tokens = 1_000_000, 100_000
            return out

    router = ModelRouter(
        Settings(_env_file=None, llm_provider="anthropic", llm_model="claude-opus-5-5"),
        client=ClaudeClient(['{"gaps": []}']),
    )
    router.listeners.append(records.append)
    router.json("gap.discovery", "s", "u", GapAnalysis)
    assert records[0].model == "claude-sonnet-5-5"
    assert abs(records[0].cost_usd - (2.0 + 1.0)) < 1e-9


def test_mock_provider_uses_context():
    router = ModelRouter(Settings(_env_file=None, llm_provider="mock", llm_model="mock"))
    context = {"papers": [{"id": "P001", "title": "t", "limitations": []}]}
    out = router.json("gap.discovery", "s", "u", GapAnalysis, context=context)
    assert out.gaps and out.gaps[0].supporting_papers == ["P001"]


def test_code_fields_are_not_requested_from_models():
    client = ScriptedClient(['{"gaps": []}'])
    ModelRouter(settings(), client=client).json("gap.discovery", "s", "u", GapAnalysis)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "search_record" not in prompt
    assert '"description"' in prompt
    assert Gap.model_fields["id"].json_schema_extra == {"llm": False}


def test_json_and_code_helpers():
    assert parse_json_object("{'a': 1}") == {"a": 1}
    assert parse_json_object("{'a': ...}") is None
    assert extract_code_block("x\n```python\nprint(1)\n```\n") == "print(1)"
