"""Task registry for difficulty-based model routing.

Routing never looks at which agent issues a call. Every model call is a registered
task kind with a difficulty and a capability; the tier and reasoning effort are
derived from those two properties. One agent therefore mixes tiers: the Literature
Intelligence agent uses the lite tier for extraction while the Gap & Novelty Analyst
uses the strong tier for novelty judgments.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Dict, Mapping


class Difficulty(IntEnum):
    LOW = 1
    LOW_MEDIUM = 2
    MEDIUM = 3
    MEDIUM_HIGH = 4
    HIGH = 5
    VERY_HIGH = 6

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: "str | Difficulty") -> "Difficulty":
        if isinstance(value, Difficulty):
            return value
        key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        return cls[key]


class Tier(str, Enum):
    CODE = "code"
    LITE = "lite"
    STANDARD = "standard"
    STRONG = "strong"
    CODING = "coding"


class Capability(str, Enum):
    REASONING = "reasoning"
    EXTRACTION = "extraction"
    WRITING = "writing"
    CODING = "coding"
    COMPUTE = "compute"


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    label: str
    difficulty: Difficulty
    capability: Capability
    issued_by: str


_TASKS = [
    TaskSpec("field.search_queries", "Field Scout: broad search query generation", Difficulty.LOW, Capability.EXTRACTION, "field_scout"),
    TaskSpec("field.map", "Field Scout: broad field map", Difficulty.LOW_MEDIUM, Capability.REASONING, "field_scout"),
    TaskSpec("literature.search_queries", "Literature retrieval queries", Difficulty.LOW, Capability.EXTRACTION, "literature"),
    TaskSpec("literature.dedup", "Literature retrieval / deduplication", Difficulty.LOW, Capability.COMPUTE, "literature"),
    TaskSpec("literature.extraction", "Paper metadata extraction and summarization", Difficulty.LOW_MEDIUM, Capability.EXTRACTION, "literature"),
    TaskSpec("literature.citation_graph", "Citation graph construction", Difficulty.LOW, Capability.COMPUTE, "literature"),
    TaskSpec("literature.clustering", "Clustering papers by methodology", Difficulty.LOW_MEDIUM, Capability.EXTRACTION, "literature"),
    TaskSpec("literature.claim_verification", "Verifying paper claims against full text", Difficulty.MEDIUM, Capability.EXTRACTION, "literature"),
    TaskSpec("gap.discovery", "Gap discovery", Difficulty.HIGH, Capability.REASONING, "gap_novelty"),
    TaskSpec("novelty.analysis", "Novelty analysis", Difficulty.VERY_HIGH, Capability.REASONING, "gap_novelty"),
    TaskSpec("critique.proposal", "Proposal criticism", Difficulty.VERY_HIGH, Capability.REASONING, "proposal_critic"),
    TaskSpec("critique.design", "Hypothesis and experiment criticism", Difficulty.VERY_HIGH, Capability.REASONING, "proposal_critic"),
    TaskSpec("proposal.revision", "Proposal revision / pivot", Difficulty.HIGH, Capability.REASONING, "planner"),
    TaskSpec("hypothesis.design", "Hypothesis design", Difficulty.VERY_HIGH, Capability.REASONING, "hypothesis_designer"),
    TaskSpec("experiment.design", "Experiment design", Difficulty.VERY_HIGH, Capability.REASONING, "experiment_designer"),
    TaskSpec("experiment.coding", "Experiment implementation", Difficulty.MEDIUM_HIGH, Capability.CODING, "experiment_engineer"),
    TaskSpec("experiment.debugging", "Experiment failure repair", Difficulty.MEDIUM_HIGH, Capability.CODING, "experiment_engineer"),
    TaskSpec("experiment.execution", "Experiment execution", Difficulty.LOW, Capability.COMPUTE, "experiment_engineer"),
    TaskSpec("data.processing", "Data processing / aggregation", Difficulty.LOW, Capability.COMPUTE, "quant_analyst"),
    TaskSpec("stats.calculation", "Statistical calculations", Difficulty.LOW_MEDIUM, Capability.COMPUTE, "quant_analyst"),
    TaskSpec("stats.summary", "Statistical result summary", Difficulty.LOW_MEDIUM, Capability.WRITING, "quant_analyst"),
    TaskSpec("science.interpretation", "Scientific interpretation", Difficulty.HIGH, Capability.REASONING, "interpreter"),
    TaskSpec("research.planning", "Research planning", Difficulty.HIGH, Capability.REASONING, "planner"),
    TaskSpec("paper.claims", "Claim-evidence mapping", Difficulty.HIGH, Capability.REASONING, "paper_architect"),
    TaskSpec("paper.drafting", "Paper drafting", Difficulty.MEDIUM_HIGH, Capability.WRITING, "paper_architect"),
    TaskSpec("citation.formatting", "Citation formatting", Difficulty.LOW, Capability.COMPUTE, "paper_architect"),
    TaskSpec("text.rewriting", "Simple rewriting (plain-language summary)", Difficulty.LOW, Capability.WRITING, "paper_architect"),
    TaskSpec("review.simulation", "Reviewer simulation", Difficulty.VERY_HIGH, Capability.REASONING, "red_team"),
    TaskSpec("json.repair", "Malformed JSON repair", Difficulty.LOW, Capability.EXTRACTION, "router"),
]

TASKS: Dict[str, TaskSpec] = {spec.kind: spec for spec in _TASKS}


def get_task(kind: str) -> TaskSpec:
    try:
        return TASKS[kind]
    except KeyError as exc:
        raise KeyError(f"Unknown task kind: {kind}") from exc


def resolve_tier(
    spec: TaskSpec,
    standard_min: Difficulty = Difficulty.MEDIUM_HIGH,
    strong_min: Difficulty = Difficulty.VERY_HIGH,
    overrides: Mapping[str, Tier] | None = None,
) -> Tier:
    override = (overrides or {}).get(spec.kind)
    if override is not None:
        return override
    if spec.capability is Capability.COMPUTE:
        return Tier.CODE
    if spec.capability is Capability.CODING:
        return Tier.CODING
    if spec.difficulty >= strong_min:
        return Tier.STRONG
    if spec.difficulty >= standard_min:
        return Tier.STANDARD
    return Tier.LITE


def effort_for(difficulty: Difficulty) -> str:
    if difficulty <= Difficulty.LOW_MEDIUM:
        return "low"
    if difficulty <= Difficulty.MEDIUM_HIGH:
        return "medium"
    return "high"


def parse_overrides(raw: str) -> Dict[str, Tier]:
    """Parse "task=tier,task=tier" into a mapping, ignoring blanks."""
    out: Dict[str, Tier] = {}
    for chunk in (raw or "").replace(";", ",").split(","):
        if "=" not in chunk:
            continue
        kind, tier = (part.strip() for part in chunk.split("=", 1))
        if not kind or not tier:
            continue
        get_task(kind)
        out[kind] = Tier(tier.lower())
    return out
