"""Typed research-state objects.

Most models double as LLM output schemas. Fields declared with ``code_field`` are owned
by deterministic code (ids, provenance, derived states) and are never requested from a
model, so a model cannot fabricate them.
"""

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_field(default: Any = None, **kwargs: Any) -> Any:
    if isinstance(default, (list, dict)):
        factory = type(default)
        return Field(default_factory=factory, json_schema_extra={"llm": False}, **kwargs)
    return Field(default, json_schema_extra={"llm": False}, **kwargs)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "statement", "description", "name", "title", "value", "id"):
            if isinstance(value.get(key), str):
                return value[key].strip()
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _to_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "\n".join(s for s in (_stringify(v) for v in value) if s)
    return _stringify(value)


def _to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        return [_stringify(value)]
    if isinstance(value, (list, tuple, set)):
        return [s for s in (_stringify(v) for v in value) if s]
    return [_stringify(value)]


_WORD_SCORES = {"very low": 0.1, "low": 0.25, "medium": 0.5, "moderate": 0.5, "high": 0.8, "very high": 0.95}


def _to_unit(value: Any) -> float:
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _WORD_SCORES:
            return _WORD_SCORES[word]
        try:
            value = float(word.rstrip("%")) / (100 if word.endswith("%") else 1)
        except ValueError:
            return 0.5
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number > 1.0 and number <= 10.0:
        number = number / 10.0
    elif number > 10.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _to_int_list(value: Any) -> List[int]:
    items = value if isinstance(value, (list, tuple)) else [value] if value not in (None, "") else []
    out: List[int] = []
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _choice(options: tuple, default: str, aliases: Dict[str, str] | None = None):
    lookup = {opt.lower(): opt for opt in options}
    lookup.update({k.lower(): v for k, v in (aliases or {}).items()})

    def normalize(value: Any) -> str:
        key = _stringify(value).lower().replace("-", "_").replace(" ", "_")
        return lookup.get(key, lookup.get(_stringify(value).lower(), default))

    return BeforeValidator(normalize)


def _priority(value: Any) -> int:
    try:
        return max(1, min(5, int(float(value))))
    except (TypeError, ValueError):
        return 3


Text = Annotated[str, BeforeValidator(_to_text)]
StrList = Annotated[List[str], BeforeValidator(_to_str_list)]
Score = Annotated[float, BeforeValidator(_to_unit)]
IntList = Annotated[List[int], BeforeValidator(_to_int_list)]
Flag = Annotated[bool, BeforeValidator(_to_bool)]
Priority = Annotated[int, BeforeValidator(_priority)]

STRENGTHS = ("none", "weak", "moderate", "strong")
Strength = Annotated[Literal[STRENGTHS], _choice(STRENGTHS, "weak", {"medium": "moderate", "high": "strong", "low": "weak"})]

EVIDENCE_STATES = ("SUPPORTED", "PARTIALLY_SUPPORTED", "HYPOTHESIS", "SPECULATION", "CONTRADICTED", "UNKNOWN")


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------- project


class Usage(Model):
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    escalations: int = 0
    by_tier: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class Project(Model):
    id: str
    title: str
    topic: str
    question: str = ""
    seed_papers: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    status: str = "created"
    phase: str = "initialization"
    outcome: str = ""
    stop_reason: str = ""
    error: str = ""
    step: int = 0
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    usage: Usage = Field(default_factory=Usage)
    completion: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- field


class Term(Model):
    term: Text = ""
    definition: Text = ""
    synonyms: StrList = Field(default_factory=list)


class FieldMap(Model):
    domains: StrList = Field(default_factory=list)
    subdomains: StrList = Field(default_factory=list)
    terminology: List[Term] = Field(default_factory=list)
    major_problems: StrList = Field(default_factory=list)
    common_methods: StrList = Field(default_factory=list)
    datasets: StrList = Field(default_factory=list)
    benchmarks: StrList = Field(default_factory=list)
    metrics: StrList = Field(default_factory=list)
    emerging_directions: StrList = Field(default_factory=list)
    active_areas: StrList = Field(default_factory=list)
    neglected_areas: StrList = Field(default_factory=list)
    open_questions: StrList = Field(default_factory=list)
    literature_queries: StrList = Field(default_factory=list, description="Targeted academic search queries for the literature review")
    key_research_groups: List[str] = code_field([])
    venues: List[str] = code_field([])
    recent_developments: List[str] = code_field([])
    provenance: str = code_field("")


class QueryPlan(Model):
    queries: StrList = Field(default_factory=list, description="Short academic search queries (3 to 8 words each)")
    synonyms: StrList = Field(default_factory=list)


# ---------------------------------------------------------------- literature


EVIDENCE_QUALITY = ("high", "medium", "low", "unknown")


class PaperClaim(Model):
    text: Text = ""
    basis: Annotated[Literal["demonstrated", "claimed"], _choice(("demonstrated", "claimed"), "claimed")] = "claimed"
    evidence: Text = Field("", description="Which experiment or analysis in the paper backs the claim, if any")


class PaperExtraction(Model):
    id: Text = Field("", description="Paper id exactly as given, e.g. P007")
    problem: Text = ""
    method: Text = ""
    novelty: Text = ""
    datasets: StrList = Field(default_factory=list)
    baselines: StrList = Field(default_factory=list)
    metrics: StrList = Field(default_factory=list)
    results: Text = ""
    limitations: StrList = Field(default_factory=list)
    assumptions: StrList = Field(default_factory=list)
    claims: List[PaperClaim] = Field(default_factory=list)
    summary: Text = Field("", description="Two sentence summary")
    relevance_to_proposal: Text = ""
    relevance_score: Score = Field(0.0, description="0 to 1 relevance to the proposal")
    evidence_quality: Annotated[Literal[EVIDENCE_QUALITY], _choice(EVIDENCE_QUALITY, "unknown")] = "unknown"


class Paper(PaperExtraction):
    title: str = code_field("")
    year: int | None = code_field(None)
    authors: List[str] = code_field([])
    venue: str = code_field("")
    abstract: str = code_field("")
    doi: str = code_field("")
    arxiv_id: str = code_field("")
    openalex_id: str = code_field("")
    s2_id: str = code_field("")
    url: str = code_field("")
    pdf_url: str = code_field("")
    cited_by_count: int = code_field(0)
    references: List[str] = code_field([])
    source: str = code_field("")
    synthetic: bool = code_field(False)
    seed: bool = code_field(False)
    found_by: List[str] = code_field([])
    text_basis: str = code_field("abstract")
    extracted: bool = code_field(False)
    cluster: str = code_field("")
    added_at: str = code_field("")


class ExtractionBatch(Model):
    papers: List[PaperExtraction] = Field(default_factory=list)


class Cluster(Model):
    name: Text = ""
    approach: Text = ""
    paper_ids: StrList = Field(default_factory=list)


class LiteratureClustering(Model):
    clusters: List[Cluster] = Field(default_factory=list)
    state_of_the_art: StrList = Field(default_factory=list, description="Paper ids representing the current state of the art")
    redundant_approaches: StrList = Field(default_factory=list)
    contradictions: StrList = Field(default_factory=list, description="Contradictory findings across papers, citing paper ids")


class ClaimCheck(Model):
    paper_id: Text = ""
    claim: Text = ""
    verdict: Annotated[
        Literal["demonstrated", "claimed_only", "not_found"],
        _choice(("demonstrated", "claimed_only", "not_found"), "claimed_only"),
    ] = "claimed_only"
    quote: Text = Field("", description="Short supporting quote from the text")


class ClaimVerification(Model):
    checks: List[ClaimCheck] = Field(default_factory=list)


class SearchRecord(Model):
    providers: List[str] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    papers_checked: int = 0
    rounds: int = 0
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------- gaps and novelty


GAP_CATEGORIES = (
    "limitation",
    "unresolved_problem",
    "contradiction",
    "missing_experiment",
    "missing_dataset",
    "unexplored_combination",
    "weak_assumption",
    "scalability",
    "generalization",
    "reproducibility",
    "evaluation_weakness",
    "underexplored_direction",
)


class Gap(Model):
    id: str = code_field("")
    description: Text = ""
    category: Annotated[Literal[GAP_CATEGORIES], _choice(GAP_CATEGORIES, "underexplored_direction")] = "underexplored_direction"
    supporting_papers: StrList = Field(default_factory=list, description="Paper ids (P003) that evidence the gap")
    evidence: Text = ""
    existing_attempts: Text = ""
    why_unsolved: Text = ""
    proposed_solution: Text = ""
    novelty_type: Text = ""
    novelty_strength: Strength = "weak"
    risks: StrList = Field(default_factory=list)
    search_record: SearchRecord | None = code_field(None)


class GapAnalysis(Model):
    gaps: List[Gap] = Field(default_factory=list)
    coverage_assessment: Text = ""
    coverage_sufficient: Flag = Field(True, description="false if important literature is clearly missing")
    missing_literature_queries: StrList = Field(default_factory=list, description="Queries to run if coverage is insufficient")


class PriorWork(Model):
    paper_id: Text = ""
    overlap: Text = ""
    difference: Text = ""


NOVELTY_VERDICTS = ("novel", "incremental", "already_done", "unclear")


class NoveltyAssessment(Model):
    proposal_version: int = code_field(0)
    closest_prior_work: List[PriorWork] = Field(default_factory=list)
    differentiation: Text = ""
    novelty_type: Text = Field("", description="new method, new combination, new application, new evaluation, or none")
    novelty_strength: Strength = "weak"
    verdict: Annotated[Literal[NOVELTY_VERDICTS], _choice(NOVELTY_VERDICTS, "unclear")] = "unclear"
    confidence: Score = 0.5
    why_not_already_covered: Text = ""
    search_record: SearchRecord | None = code_field(None)
    warnings: List[str] = code_field([])
    created_at: str = code_field("")


# ---------------------------------------------------------------- proposal and critique


class ProposalVersion(Model):
    version: int
    text: str
    rationale: str = ""
    decision_id: str = ""
    created_at: str = Field(default_factory=now_iso)


class ProposalRevision(Model):
    action: Annotated[Literal["revise", "pivot"], _choice(("revise", "pivot"), "revise")] = "revise"
    revised_proposal: Text = ""
    rationale: Text = ""
    changes: StrList = Field(default_factory=list)


CRITICISM_CATEGORIES = (
    "weakness",
    "hidden_assumption",
    "novelty_risk",
    "methodological_risk",
    "evaluation_risk",
    "reviewer_objection",
    "leakage_risk",
    "confounder",
    "baseline_gap",
    "generalization",
)
SEVERITIES = ("fatal", "major", "minor")


class Criticism(Model):
    id: str = code_field("")
    category: Annotated[Literal[CRITICISM_CATEGORIES], _choice(CRITICISM_CATEGORIES, "weakness")] = "weakness"
    severity: Annotated[Literal[SEVERITIES], _choice(SEVERITIES, "major", {"critical": "fatal", "high": "major", "low": "minor"})] = "major"
    target_id: Text = Field("", description="Id of the hypothesis (H1) or experiment (E2) criticised, if any")
    description: Text = ""
    proposed_test: Text = Field("", description="Test or evidence that would resolve this criticism")
    status: str = code_field("open")


VERDICTS = ("proceed", "revise", "pivot", "kill")


class Assessment(Model):
    id: str = code_field("")
    target: str = code_field("")
    strengths: StrList = Field(default_factory=list)
    issues: List[Criticism] = Field(default_factory=list)
    required_evidence: StrList = Field(default_factory=list)
    verdict: Annotated[Literal[VERDICTS], _choice(VERDICTS, "proceed")] = "proceed"
    summary: Text = ""
    created_at: str = code_field("")


# ---------------------------------------------------------------- hypotheses and experiments


HYPOTHESIS_STATUSES = ("active", "supported", "refuted", "inconclusive", "abandoned", "superseded")


class Hypothesis(Model):
    id: Text = Field("", description="H1, H2 ... for new hypotheses; the existing id when revising")
    version: int = code_field(1)
    statement: Text = ""
    rationale: Text = ""
    independent_variables: StrList = Field(default_factory=list)
    dependent_variables: StrList = Field(default_factory=list)
    controls: StrList = Field(default_factory=list)
    expected_outcome: Text = ""
    falsification_criteria: Text = ""
    competing_hypotheses: StrList = Field(default_factory=list)
    assumptions: StrList = Field(default_factory=list)
    motivated_by: StrList = Field(default_factory=list, description="Gap ids (G1) and paper ids (P003) motivating this hypothesis")
    priority: Priority = Field(3, description="1 (highest) to 5")
    status: str = code_field("active")
    revision_of: str = code_field("")
    revision_reason: str = code_field("")
    triggering_evidence: List[str] = code_field([])
    created_at: str = code_field("")

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"


class HypothesisSet(Model):
    question_decomposition: StrList = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)


class MetricSpec(Model):
    name: Text = ""
    higher_is_better: Flag = True
    primary: Flag = False
    min_effect: float | None = Field(None, description="Smallest difference that matters in practice, or null")


EXPERIMENT_KINDS = ("primary", "ablation", "robustness", "sensitivity", "baseline_reproduction", "validation")


class ExperimentSpec(Model):
    id: Text = Field("", description="Empty for a new experiment; the existing id (E1) when revising")
    version: int = code_field(1)
    hypothesis_id: Text = ""
    hypothesis_version: int = code_field(0)
    kind: Annotated[Literal[EXPERIMENT_KINDS], _choice(EXPERIMENT_KINDS, "primary")] = "primary"
    objective: Text = ""
    datasets: StrList = Field(default_factory=list)
    method: Text = Field("", description="Arm name of the proposed method")
    baselines: StrList = Field(default_factory=list, description="Arm names of the baselines")
    controls: StrList = Field(default_factory=list)
    metrics: List[MetricSpec] = Field(default_factory=list)
    ablations: StrList = Field(default_factory=list)
    seeds: IntList = Field(default_factory=list)
    repetitions: int = 1
    expected_outcomes: Text = ""
    failure_criteria: Text = ""
    statistical_plan: Text = ""
    confounders: StrList = Field(default_factory=list)
    robustness_tests: StrList = Field(default_factory=list)
    sensitivity_analyses: StrList = Field(default_factory=list)
    compute_budget: Text = ""
    status: str = code_field("draft")
    change_log: List[str] = code_field([])
    critique_notes: List[str] = code_field([])
    created_at: str = code_field("")

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"

    @property
    def arms(self) -> List[str]:
        return [self.method, *self.baselines]

    def primary_metric(self) -> MetricSpec | None:
        for metric in self.metrics:
            if metric.primary:
                return metric
        return self.metrics[0] if self.metrics else None


class ExperimentPlan(Model):
    experiments: List[ExperimentSpec] = Field(default_factory=list)
    notes: Text = ""


class ImplementationNotes(Model):
    dataset_version: Text = ""
    model_version: Text = ""
    compute_estimate: Text = ""
    assumptions: StrList = Field(default_factory=list)
    deviations: StrList = Field(default_factory=list, description="Any unavoidable departure from the spec")


class RunRecord(Model):
    arm: str
    seed: int
    metrics: Dict[str, float] = Field(default_factory=dict)


class RunAttempt(Model):
    attempt: int
    code_version: str = ""
    status: str = ""
    error: str = ""
    duration_s: float = 0.0
    finished_at: str = Field(default_factory=now_iso)


class ExperimentRun(Model):
    id: str
    experiment_id: str
    spec_version: int
    hypothesis_id: str = ""
    code_version: str = ""
    dataset_version: str = ""
    model_version: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)
    seeds: List[int] = Field(default_factory=list)
    executor: str = ""
    code_generator: str = ""
    compute: str = ""
    duration_s: float = 0.0
    status: str = "awaiting_execution"
    synthetic: bool = False
    records: List[RunRecord] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    deviations: List[str] = Field(default_factory=list)
    attempts: List[RunAttempt] = Field(default_factory=list)
    error: str = ""
    created_at: str = Field(default_factory=now_iso)
    completed_at: str = ""

    @property
    def spec_key(self) -> str:
        return f"{self.experiment_id}@v{self.spec_version}"


# ---------------------------------------------------------------- analysis


class ArmSummary(Model):
    metric: str
    arm: str
    n: int
    mean: float
    std: float
    ci_low: float | None = None
    ci_high: float | None = None
    min: float
    max: float


class Comparison(Model):
    metric: str
    method: str
    baseline: str
    higher_is_better: bool = True
    primary: bool = False
    diff: float
    rel_diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    t: float | None = None
    df: float | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    hedges_g: float | None = None
    significant: bool = False
    direction_ok: bool = False
    practically_significant: bool = False
    stable_without_best_seed: bool | None = None


class StatsSummary(Model):
    summary: Text = ""
    interpretation_constraints: StrList = Field(default_factory=list)


class QuantResult(Model):
    run_id: str
    experiment_id: str
    hypothesis_id: str = ""
    alpha: float = 0.05
    synthetic: bool = False
    summaries: List[ArmSummary] = Field(default_factory=list)
    comparisons: List[Comparison] = Field(default_factory=list)
    anomalies: List[str] = Field(default_factory=list)
    interpretation_constraints: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: str = Field(default_factory=now_iso)


STATEMENT_TAGS = ("evidence_backed", "inference", "hypothesis", "speculation")


class Statement(Model):
    text: Text = ""
    tag: Annotated[Literal[STATEMENT_TAGS], _choice(STATEMENT_TAGS, "inference", {"evidence-backed": "evidence_backed", "fact": "evidence_backed"})] = "inference"
    evidence_refs: StrList = Field(default_factory=list, description="Run ids (R001) or paper ids (P003) backing the statement")


class Explanation(Model):
    explanation: Text = ""
    plausibility: Score = 0.5
    follow_up: Text = Field("", description="Experiment or analysis that would discriminate this explanation")


class FollowUp(Model):
    statement: Text = ""
    rationale: Text = ""


INTERPRETATION_VERDICTS = ("supported", "refuted", "inconclusive")


class Interpretation(Model):
    run_id: str = code_field("")
    hypothesis_id: str = code_field("")
    verdict: Annotated[Literal[INTERPRETATION_VERDICTS], _choice(INTERPRETATION_VERDICTS, "inconclusive")] = "inconclusive"
    verdict_rationale: Text = ""
    statements: List[Statement] = Field(default_factory=list)
    competing_explanations: List[Explanation] = Field(default_factory=list)
    unexpected_findings: StrList = Field(default_factory=list)
    contradicting_evidence: StrList = Field(default_factory=list)
    follow_up_hypotheses: List[FollowUp] = Field(default_factory=list)
    mechanism_supported: Flag = False
    adjustments: List[str] = code_field([])
    created_at: str = code_field("")


# ---------------------------------------------------------------- planning and decisions


PLANNER_ACTIONS = (
    "SEARCH_LITERATURE",
    "RUN_EXPERIMENT",
    "REPRODUCE_BASELINE",
    "RUN_ABLATION",
    "RUN_ROBUSTNESS_TEST",
    "INVESTIGATE_FAILURE",
    "REVISE_HYPOTHESIS",
    "FORM_HYPOTHESIS",
    "DESIGN_EXPERIMENT",
    "REVISE_PROPOSAL",
    "WRITE_PAPER",
    "REQUEST_REVIEW",
    "STOP",
    "KILL",
)
LIFECYCLE_ACTIONS = (
    "SCOUT_FIELD",
    "ANALYZE_RUN",
    "ANALYZE_GAPS",
    "ASSESS_NOVELTY",
    "CRITIQUE_PROPOSAL",
    "CRITIQUE_DESIGN",
    "PLAN",
)
ACTION_TYPES = PLANNER_ACTIONS + LIFECYCLE_ACTIONS


class Action(Model):
    id: str = code_field("")
    type: Annotated[str, BeforeValidator(lambda v: _stringify(v).upper().replace(" ", "_").replace("-", "_"))] = ""
    target: Text = Field("", description="Id the action applies to (H1, E2, R003, RT1-2) or a search query")
    rationale: Text = ""
    scientific_value: Score = 0.5
    uncertainty_reduction: Score = 0.5
    feasibility: Score = 0.5
    cost: Score = Field(0.5, description="Relative cost 0 (free) to 1 (very expensive)")
    addresses: StrList = Field(default_factory=list, description="Issue, criticism or claim ids this action resolves")
    priority: float = code_field(0.0)
    origin: str = code_field("planner")
    status: str = code_field("pending")
    params: Dict[str, Any] = code_field({})
    result: str = code_field("")
    created_at: str = code_field("")
    finished_at: str = code_field("")


class AbandonRequest(Model):
    hypothesis_id: Text = ""
    reason: Text = ""


class PlannerOutput(Model):
    assessment: Text = ""
    evidence_sufficient_for_paper: Flag = False
    candidates: List[Action] = Field(default_factory=list)
    abandon: List[AbandonRequest] = Field(default_factory=list)
    blockers: StrList = Field(default_factory=list)


class Decision(Model):
    id: str
    timestamp: str = Field(default_factory=now_iso)
    kind: str
    decision: str
    reason: str = ""
    evidence: List[str] = Field(default_factory=list)
    alternatives_considered: List[str] = Field(default_factory=list)
    agent_inputs: List[str] = Field(default_factory=list)
    downstream_effects: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------- paper and review


CLAIM_KINDS = ("contribution", "finding", "background", "limitation")


class ManuscriptClaim(Model):
    id: Text = Field("", description="C1, C2 ...")
    text: Text = ""
    kind: Annotated[Literal[CLAIM_KINDS], _choice(CLAIM_KINDS, "finding")] = "finding"
    importance: Annotated[Literal["central", "supporting"], _choice(("central", "supporting"), "supporting")] = "supporting"
    hypothesis_ids: StrList = Field(default_factory=list)
    run_ids: StrList = Field(default_factory=list)
    paper_ids: StrList = Field(default_factory=list)
    sections: StrList = Field(default_factory=list, description="Manuscript sections where the claim is stated")
    state: str = code_field("UNKNOWN")
    evidence_note: str = code_field("")


class ClaimMap(Model):
    storyline: Text = ""
    claims: List[ManuscriptClaim] = Field(default_factory=list)


MANUSCRIPT_SECTIONS = (
    "introduction",
    "related_work",
    "method",
    "experimental_setup",
    "results",
    "discussion",
    "limitations",
    "conclusion",
)


class Draft(Model):
    title: Text = ""
    abstract: Text = ""
    introduction: Text = ""
    related_work: Text = ""
    method: Text = ""
    experimental_setup: Text = ""
    results: Text = ""
    discussion: Text = ""
    limitations: Text = ""
    conclusion: Text = ""


REVIEW_DIMENSIONS = (
    "novelty",
    "significance",
    "methodology",
    "experiments",
    "baselines",
    "statistics",
    "reproducibility",
    "claims",
    "writing",
)
REVIEW_ACTIONS = ("experiment", "literature", "analysis", "revision", "none")


class ReviewIssue(Model):
    id: str = code_field("")
    severity: Annotated[Literal["critical", "major", "minor"], _choice(("critical", "major", "minor"), "major", {"fatal": "critical"})] = "major"
    dimension: Annotated[Literal[REVIEW_DIMENSIONS], _choice(REVIEW_DIMENSIONS, "claims")] = "claims"
    description: Text = ""
    affected_claims: StrList = Field(default_factory=list)
    required_action: Annotated[Literal[REVIEW_ACTIONS], _choice(REVIEW_ACTIONS, "revision")] = "revision"
    suggestion: Text = ""
    status: str = code_field("open")
    round: int = code_field(0)


RECOMMENDATIONS = ("accept", "minor_revision", "major_revision", "reject")


class ReviewReport(Model):
    round: int = code_field(0)
    recommendation: Annotated[Literal[RECOMMENDATIONS], _choice(RECOMMENDATIONS, "major_revision", {"minor": "minor_revision", "major": "major_revision"})] = "major_revision"
    summary: Text = ""
    issues: List[ReviewIssue] = Field(default_factory=list)
    missing_experiments: StrList = Field(default_factory=list)
    unsupported_claims: StrList = Field(default_factory=list)
    novelty_concerns: StrList = Field(default_factory=list)
    methodological_concerns: StrList = Field(default_factory=list)
    reproducibility_concerns: StrList = Field(default_factory=list)
    likely_reviewer_questions: StrList = Field(default_factory=list)
    resolved_issue_ids: StrList = Field(default_factory=list, description="Ids of previously raised issues that are now resolved")
    created_at: str = code_field("")


# ---------------------------------------------------------------- agent protocol


class TaskEnvelope(Model):
    task_id: str
    type: str
    agent: str
    objective: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    required_output: str = ""
    created_at: str = Field(default_factory=now_iso)


class AgentResult(Model):
    task_id: str
    agent: str
    status: str = "ok"
    findings: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    follow_up_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    duration_s: float = 0.0
    error: str = ""
