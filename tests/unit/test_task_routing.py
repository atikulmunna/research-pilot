import pytest

from research_pilot.config import Settings
from research_pilot.llm.router import ModelRouter
from research_pilot.llm.tasks import TASKS, Capability, Difficulty, Tier, effort_for, parse_overrides, resolve_tier

# The routing table from the redesign brief: task -> expected tier.
BRIEF = {
    "field.map": Tier.LITE,
    "literature.dedup": Tier.CODE,
    "literature.extraction": Tier.LITE,
    "literature.citation_graph": Tier.CODE,
    "gap.discovery": Tier.STRONG,
    "novelty.analysis": Tier.STRONG,
    "critique.proposal": Tier.STRONG,
    "hypothesis.design": Tier.STRONG,
    "experiment.design": Tier.STRONG,
    "experiment.coding": Tier.CODING,
    "data.processing": Tier.CODE,
    "stats.calculation": Tier.CODE,
    "science.interpretation": Tier.STRONG,
    "research.planning": Tier.STRONG,
    "paper.drafting": Tier.STRONG,
    "citation.formatting": Tier.CODE,
    "review.simulation": Tier.STRONG,
    "text.rewriting": Tier.LITE,
}


def _settings(**kw):
    return Settings(_env_file=None, llm_provider="openrouter", llm_model="default-model", **kw)


@pytest.mark.parametrize("kind,tier", sorted(BRIEF.items()))
def test_brief_routing_table(kind, tier):
    assert resolve_tier(TASKS[kind]) is tier


def test_routing_ignores_issuing_agent():
    # The same agent issues tasks on different tiers.
    literature = {k: resolve_tier(s) for k, s in TASKS.items() if s.issued_by == "literature"}
    assert {Tier.LITE, Tier.CODE} <= set(literature.values())
    quant = {resolve_tier(s) for s in TASKS.values() if s.issued_by == "quant_analyst"}
    assert quant == {Tier.CODE, Tier.LITE}


def test_effort_follows_difficulty():
    assert effort_for(Difficulty.LOW) == "low"
    assert effort_for(Difficulty.MEDIUM_HIGH) == "medium"
    assert effort_for(Difficulty.VERY_HIGH) == "high"


def test_tiers_fall_back_to_default_model():
    router = ModelRouter(_settings())
    for kind in ("field.map", "novelty.analysis", "experiment.coding"):
        route = router.resolve(kind)
        assert (route.provider, route.model) == ("openrouter", "default-model")


def test_tier_specific_models_and_coding_fallback_to_strong():
    router = ModelRouter(
        _settings(llm_lite_provider="anthropic", llm_lite_model="claude-haiku-4-5", llm_strong_provider="anthropic", llm_strong_model="claude-opus-5-5")
    )
    lite = router.resolve("literature.extraction")
    assert (lite.provider, lite.model) == ("anthropic", "claude-haiku-4-5")
    assert router.resolve("novelty.analysis").model == "claude-opus-5-5"
    coding = router.resolve("experiment.coding")
    assert (coding.provider, coding.model, coding.tier) == ("anthropic", "claude-opus-5-5", Tier.CODING)


def test_anthropic_defaults_route_lite_work_to_haiku():
    router = ModelRouter(Settings(_env_file=None))
    assert (router.resolve("field.map").provider, router.resolve("field.map").model) == ("anthropic", "claude-haiku-4-5")
    assert router.resolve("review.simulation").model == "claude-opus-5-5"
    assert router.resolve("experiment.coding").model == "claude-opus-5-5"


def test_reasoning_effort_modes():
    router = ModelRouter(_settings(llm_strong_reasoning_effort="medium", llm_lite_reasoning_effort="none"))
    assert router.resolve("review.simulation").reasoning_effort == "medium"
    assert router.resolve("field.map").reasoning_effort is None
    auto = ModelRouter(_settings())
    assert auto.resolve("review.simulation").reasoning_effort == "high"
    assert auto.resolve("field.search_queries").reasoning_effort == "low"


def test_threshold_and_overrides():
    router = ModelRouter(_settings(llm_strong_min_difficulty="very_high", llm_task_overrides="paper.drafting=lite"))
    assert router.resolve("gap.discovery").tier is Tier.LITE
    assert router.resolve("novelty.analysis").tier is Tier.STRONG
    assert router.resolve("paper.drafting").tier is Tier.LITE


def test_override_rejects_unknown_task():
    with pytest.raises(KeyError):
        parse_overrides("not.a.task=lite")


def test_every_task_is_routable():
    table = ModelRouter(_settings()).routing_table()
    assert len(table) == len(TASKS)
    assert all(row["tier"] == "code" for row in table if TASKS[row["task"]].capability is Capability.COMPUTE)
