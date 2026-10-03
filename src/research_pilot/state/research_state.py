"""Guarded access to the shared research state.

Agents read and modify the research state only through this class. It is the single
choke point for the failure-safety rules of the plan:

- No fabricated evidence: papers must come from a literature provider with a traceable
  identifier, and references to unknown papers, hypotheses, experiments or runs are
  stripped and logged as violations.
- No silent protocol changes: experiment changes go through the versioned registry.
- No retrospective hypothesis rewriting: revisions create a new version that records
  the reason and triggering evidence; the original stays visible.
- No unsupported claims: claim states are derived from the evidence graph.
"""

import re
from typing import Any, Dict, Iterable, List, Set

from ..config import Settings
from .evidence import EvidenceGraph
from .models import (
    Action,
    Assessment,
    ClaimMap,
    Criticism,
    Decision,
    ExperimentRun,
    ExperimentSpec,
    Gap,
    Hypothesis,
    NoveltyAssessment,
    Paper,
    ProposalVersion,
    now_iso,
)
from .registry import ExperimentRegistry
from .store import ProjectStore

PAPER_SOURCES = {"openalex", "semantic_scholar", "arxiv", "mock"}
LEADING_ID = re.compile(r"[A-Z]{1,2}\d+(?:-\d+)?(?:@v\d+)?(?![\w@])")


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def paper_keys(paper: Paper) -> Set[str]:
    keys = set()
    if paper.doi:
        keys.add("doi:" + paper.doi.lower().replace("https://doi.org/", "").strip())
    if paper.arxiv_id:
        keys.add("arxiv:" + re.sub(r"v\d+$", "", paper.arxiv_id.lower()))
    if paper.openalex_id:
        keys.add("openalex:" + paper.openalex_id.rsplit("/", 1)[-1].lower())
    if paper.s2_id:
        keys.add("s2:" + paper.s2_id.lower())
    title = normalize_title(paper.title)
    if len(title) > 12:
        keys.add("title:" + title)
    return keys


class ResearchState:
    def __init__(self, store: ProjectStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.graph = EvidenceGraph.from_dict(store.graph_data())
        self.experiments = ExperimentRegistry(store)
        self.violations: List[str] = []

    # ------------------------------------------------------------ bookkeeping

    def save_graph(self) -> None:
        self.store.save_graph_data(self.graph.to_dict())

    def violation(self, rule: str, detail: str) -> None:
        message = f"[{rule}] {detail}"
        self.violations.append(message)
        self.store.append_jsonl("logs/violations.jsonl", {"timestamp": now_iso(), "rule": rule, "detail": detail})

    def drain_violations(self) -> List[str]:
        out, self.violations = self.violations, []
        return out

    def filter_refs(self, refs: Iterable[str], allowed: Set[str], context: str) -> List[str]:
        out: List[str] = []
        for raw in refs:
            ref = (raw or "").strip().strip("[]{}")
            if not ref:
                continue
            # Models sometimes decorate an id ("P042: Title - note"); keep the id when it is known.
            leading = LEADING_ID.match(ref)
            ref = ref if ref in allowed or not leading else leading.group(0)
            if ref in allowed:
                if ref not in out:
                    out.append(ref)
            else:
                self.violation("no_fabricated_evidence", f"{context}: dropped unknown reference '{ref[:120]}'")
        return out

    # ------------------------------------------------------------ papers

    def papers(self) -> List[Paper]:
        return self.store.papers()

    def paper_ids(self) -> Set[str]:
        return {p.id for p in self.papers()}

    def check_paper_refs(self, refs: Iterable[str], context: str) -> List[str]:
        return self.filter_refs(refs, self.paper_ids(), context)

    def add_papers(self, candidates: Iterable[Paper], found_by: str) -> List[Paper]:
        existing = self.papers()
        index: Dict[str, Paper] = {}
        for paper in existing:
            for key in paper_keys(paper):
                index[key] = paper
        added: List[Paper] = []
        for candidate in candidates:
            traceable = any([candidate.doi, candidate.arxiv_id, candidate.openalex_id, candidate.s2_id, candidate.url])
            if candidate.source not in PAPER_SOURCES or not traceable or not candidate.title.strip():
                self.violation("no_fabricated_evidence", f"rejected paper without a traceable source: {candidate.title[:80]!r}")
                continue
            keys = paper_keys(candidate)
            match = next((index[k] for k in keys if k in index), None)
            if match is not None:
                changed = False
                for attr in ("doi", "arxiv_id", "openalex_id", "s2_id", "pdf_url", "abstract", "venue"):
                    if not getattr(match, attr) and getattr(candidate, attr):
                        setattr(match, attr, getattr(candidate, attr))
                        changed = True
                for tag in candidate.found_by or [found_by]:
                    if tag not in match.found_by:
                        match.found_by.append(tag)
                        changed = True
                if candidate.seed and not match.seed:
                    match.seed = True
                    changed = True
                if changed:
                    self.store.save_paper(match)
                continue
            paper = candidate.model_copy(deep=True)
            paper.id = self.store.next_id("P", width=3)
            paper.added_at = now_iso()
            paper.found_by = list(candidate.found_by) or [found_by]
            self.store.save_paper(paper)
            for key in keys:
                index[key] = paper
            added.append(paper)
            self.graph.add_node(
                paper.id,
                "paper",
                paper.title,
                year=paper.year,
                source=paper.source,
                synthetic=paper.synthetic,
                evidence_quality=paper.evidence_quality,
            )
        self.save_graph()
        return added

    def find_paper(self, candidate: Paper) -> Paper | None:
        keys = paper_keys(candidate)
        for paper in self.papers():
            if keys & paper_keys(paper):
                return paper
        return None

    def update_paper(self, paper: Paper) -> None:
        self.store.save_paper(paper)
        self.graph.add_node(paper.id, "paper", paper.title, evidence_quality=paper.evidence_quality, cluster=paper.cluster)
        for idx, claim in enumerate(paper.claims[:3], start=1):
            if not claim.text:
                continue
            claim_id = f"LC-{paper.id}-{idx}"
            self.graph.add_node(claim_id, "claim", claim.text[:200], kind="literature", synthetic=paper.synthetic)
            self.graph.add_edge(paper.id, claim_id, "supports", "literature.extraction", basis=claim.basis)

    def add_citation(self, citing_id: str, cited_id: str) -> None:
        if citing_id in self.graph.nodes and cited_id in self.graph.nodes and citing_id != cited_id:
            self.graph.add_edge(citing_id, cited_id, "cites", "literature.citation_graph")

    # ------------------------------------------------------------ gaps

    def gaps(self) -> List[Gap]:
        return self.store.gaps()

    def set_gaps(self, gaps: List[Gap], meta: Dict[str, Any]) -> List[Gap]:
        for node in self.graph.nodes_of("gap"):
            node.attrs["status"] = "superseded"
        stored: List[Gap] = []
        for gap in gaps:
            if not gap.description:
                continue
            gap = gap.model_copy(deep=True)
            gap.id = self.store.next_id("G")
            gap.supporting_papers = self.check_paper_refs(gap.supporting_papers, f"gap {gap.id}")
            self.graph.add_node(gap.id, "gap", gap.description[:200], category=gap.category, novelty_strength=gap.novelty_strength)
            for paper_id in gap.supporting_papers:
                self.graph.add_edge(paper_id, gap.id, "supports", "gap.discovery")
            stored.append(gap)
        self.store.save_gaps(stored, meta)
        self.save_graph()
        return stored

    # ------------------------------------------------------------ proposal, novelty, critique

    def proposal(self) -> ProposalVersion | None:
        versions = self.store.proposal_versions()
        return versions[-1] if versions else None

    def add_proposal_version(self, text: str, rationale: str = "", decision_id: str = "") -> ProposalVersion:
        versions = self.store.proposal_versions()
        version = ProposalVersion(version=len(versions) + 1, text=text, rationale=rationale, decision_id=decision_id)
        versions.append(version)
        self.store.save_proposal_versions(versions)
        return version

    def add_novelty(self, assessment: NoveltyAssessment) -> NoveltyAssessment:
        assessment = assessment.model_copy(deep=True)
        kept = []
        for prior in assessment.closest_prior_work:
            refs = self.check_paper_refs([prior.paper_id], "novelty closest_prior_work")
            if refs:
                prior.paper_id = refs[0]
                kept.append(prior)
        assessment.closest_prior_work = kept
        assessment.created_at = now_iso()
        items = self.store.novelty_analyses()
        items.append(assessment)
        self.store.save_novelty_analyses(items)
        return assessment

    def latest_novelty(self) -> NoveltyAssessment | None:
        items = self.store.novelty_analyses()
        return items[-1] if items else None

    def add_assessment(self, assessment: Assessment, target: str) -> Assessment:
        assessment = assessment.model_copy(deep=True)
        assessment.id = self.store.next_id("CR")
        assessment.target = target
        assessment.created_at = now_iso()
        known = {h.id for h in self.current_hypotheses()} | {s.id for s in self.experiments.latest_specs()}
        for idx, issue in enumerate(assessment.issues, start=1):
            issue.id = f"{assessment.id}-{idx}"
            if issue.target_id and issue.target_id not in known:
                issue.target_id = ""
        items = self.store.assessments()
        items.append(assessment)
        self.store.save_assessments(items)
        return assessment

    def open_criticisms(self, target_prefix: str = "") -> List[Criticism]:
        out = []
        for assessment in self.store.assessments():
            if target_prefix and not assessment.target.startswith(target_prefix):
                continue
            out.extend(issue for issue in assessment.issues if issue.status == "open")
        return out

    def resolve_criticisms(self, ids: Iterable[str], status: str = "addressed") -> None:
        wanted = set(ids)
        if not wanted:
            return
        items = self.store.assessments()
        for assessment in items:
            for issue in assessment.issues:
                if issue.id in wanted:
                    issue.status = status
        self.store.save_assessments(items)

    # ------------------------------------------------------------ hypotheses

    def hypotheses(self) -> List[Hypothesis]:
        return self.store.hypotheses()

    def current_hypotheses(self, include_closed: bool = True) -> List[Hypothesis]:
        latest: Dict[str, Hypothesis] = {}
        for hyp in self.store.hypotheses():
            if hyp.id not in latest or hyp.version > latest[hyp.id].version:
                latest[hyp.id] = hyp
        items = [h for h in latest.values() if h.status != "superseded"]
        if not include_closed:
            items = [h for h in items if h.status not in {"abandoned"}]
        return sorted(items, key=lambda h: (h.priority, int(h.id[1:]) if h.id[1:].isdigit() else 0))

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        hid = hypothesis_id.split("@")[0]
        versions = [h for h in self.store.hypotheses() if h.id == hid]
        return max(versions, key=lambda h: h.version) if versions else None

    def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        hyp = hypothesis.model_copy(deep=True)
        hyp.id = self.store.next_id("H")
        hyp.version = 1
        hyp.status = "active"
        hyp.created_at = now_iso()
        allowed = {g.id for g in self.gaps()} | self.paper_ids()
        hyp.motivated_by = self.filter_refs(hyp.motivated_by, allowed, f"hypothesis {hyp.id}")
        items = self.store.hypotheses()
        items.append(hyp)
        self.store.save_hypotheses(items)
        self._graph_hypothesis(hyp)
        return hyp

    def revise_hypothesis(
        self,
        hypothesis_id: str,
        revised: Hypothesis,
        reason: str,
        triggering_evidence: List[str],
        decision_id: str,
    ) -> Hypothesis:
        current = self.hypothesis(hypothesis_id)
        if current is None:
            raise KeyError(f"Unknown hypothesis: {hypothesis_id}")
        new = revised.model_copy(deep=True)
        new.id = current.id
        new.version = current.version + 1
        new.status = "active"
        new.revision_of = current.key
        new.revision_reason = f"{reason} (decision {decision_id})" if decision_id else reason
        new.triggering_evidence = list(triggering_evidence)
        new.created_at = now_iso()
        allowed = {g.id for g in self.gaps()} | self.paper_ids()
        new.motivated_by = self.filter_refs(new.motivated_by or current.motivated_by, allowed, f"hypothesis {new.key}")
        items = self.store.hypotheses()
        for item in items:
            if item.id == current.id and item.version == current.version:
                item.status = "superseded"
        items.append(new)
        self.store.save_hypotheses(items)
        for node_id in (current.key, f"C{current.key}"):
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id].attrs["status"] = "superseded"
        self._graph_hypothesis(new)
        for ref in triggering_evidence:
            if ref in self.graph.nodes and self.graph.nodes[ref].type == "result":
                self.graph.add_edge(ref, new.key, "suggests", "hypothesis.revision")
        self.save_graph()
        return new

    def set_hypothesis_status(self, hypothesis_id: str, status: str) -> None:
        current = self.hypothesis(hypothesis_id)
        if current is None:
            return
        items = self.store.hypotheses()
        for item in items:
            if item.id == current.id and item.version == current.version:
                item.status = status
        self.store.save_hypotheses(items)
        for node_id in (current.key, f"C{current.key}"):
            if node_id in self.graph.nodes:
                self.graph.nodes[node_id].attrs["status"] = status
        self.save_graph()

    def _graph_hypothesis(self, hyp: Hypothesis) -> None:
        claim_id = f"C{hyp.key}"
        self.graph.add_node(hyp.key, "hypothesis", hyp.statement[:200], status=hyp.status, priority=hyp.priority)
        self.graph.add_node(
            claim_id,
            "claim",
            hyp.statement[:200],
            kind="contribution",
            hypothesis_claim=True,
            importance="central" if hyp.priority <= 2 else "supporting",
            status=hyp.status,
        )
        self.graph.add_edge(claim_id, hyp.key, "motivates", "hypothesis.design")
        for ref in hyp.motivated_by:
            if ref in self.graph.nodes:
                self.graph.add_edge(ref, hyp.key, "motivates", "hypothesis.design")
        self.save_graph()

    # ------------------------------------------------------------ experiments

    def add_experiment(self, spec: ExperimentSpec) -> ExperimentSpec | None:
        hyp = self.hypothesis(spec.hypothesis_id)
        if hyp is None or hyp.status in {"superseded", "abandoned"}:
            self.violation("no_fabricated_evidence", f"experiment for unknown or closed hypothesis '{spec.hypothesis_id}' dropped")
            return None
        spec = spec.model_copy(update={"hypothesis_id": hyp.id})
        if not spec.seeds:
            spec.seeds = list(range(self.settings.default_seeds))
        stored = self.experiments.add_spec(spec, hypothesis_version=hyp.version)
        self.graph.add_node(stored.key, "experiment", stored.objective[:200], kind=stored.kind, status=stored.status)
        self.graph.add_edge(hyp.key, stored.key, "tested_by", "experiment.design")
        self.save_graph()
        return stored

    def revise_experiment(self, exp_id: str, revised: ExperimentSpec, reason: str) -> ExperimentSpec:
        previous = self.experiments.spec(exp_id)
        if not revised.seeds and previous is not None:
            revised = revised.model_copy(update={"seeds": previous.seeds})
        new = self.experiments.revise_spec(exp_id, revised, reason)
        if previous is not None and previous.key in self.graph.nodes:
            self.graph.nodes[previous.key].attrs["status"] = "superseded"
        hyp_key = f"{new.hypothesis_id}@v{new.hypothesis_version}"
        self.graph.add_node(new.key, "experiment", new.objective[:200], kind=new.kind, status=new.status)
        if hyp_key in self.graph.nodes:
            self.graph.add_edge(hyp_key, new.key, "tested_by", "experiment.design")
        self.save_graph()
        return new

    def approve_experiment(self, ref: str, notes: List[str] | None = None) -> ExperimentSpec:
        spec = self.experiments.approve(ref, notes)
        if spec.key in self.graph.nodes:
            self.graph.nodes[spec.key].attrs["status"] = "approved"
            self.save_graph()
        return spec

    def record_result(self, run: ExperimentRun, verdict: str) -> None:
        spec = self.experiments.spec(run.spec_key)
        self.graph.add_node(run.id, "result", f"{run.spec_key} result", synthetic=run.synthetic, verdict=verdict)
        self.graph.add_edge(run.spec_key, run.id, "produces", "experiment.execution")
        if spec is None:
            self.save_graph()
            return
        claim_id = f"C{spec.hypothesis_id}@v{spec.hypothesis_version}"
        if claim_id in self.graph.nodes:
            if verdict == "supported":
                self.graph.add_edge(run.id, claim_id, "supports", "science.interpretation", synthetic=run.synthetic)
            elif verdict == "refuted":
                self.graph.add_edge(run.id, claim_id, "refutes", "science.interpretation", synthetic=run.synthetic)
        self.save_graph()

    def claim_state(self, claim_id: str) -> str:
        return self.graph.derive_state(claim_id, self.settings.allow_synthetic_evidence)[0]

    # ------------------------------------------------------------ manuscript claims

    def set_manuscript_claims(self, claim_map: ClaimMap) -> ClaimMap:
        for node in self.graph.nodes_of("claim"):
            if node.attrs.get("manuscript"):
                node.attrs["status"] = "superseded"
        for node in self.graph.nodes_of("statement"):
            node.attrs["status"] = "superseded"
        hyp_ids = {h.id for h in self.current_hypotheses()}
        run_ids = {r.id for r in self.experiments.runs()}
        paper_ids = self.paper_ids()
        result = claim_map.model_copy(deep=True)
        kept = []
        for claim in result.claims:
            if not claim.text:
                continue
            claim.id = self.store.next_id("C")
            claim.hypothesis_ids = self.filter_refs(claim.hypothesis_ids, hyp_ids, f"claim {claim.id}")
            claim.run_ids = self.filter_refs(claim.run_ids, run_ids, f"claim {claim.id}")
            claim.paper_ids = self.filter_refs(claim.paper_ids, paper_ids, f"claim {claim.id}")
            self.graph.add_node(claim.id, "claim", claim.text[:200], kind=claim.kind, importance=claim.importance, manuscript=True)
            for hid in claim.hypothesis_ids:
                hyp = self.hypothesis(hid)
                if hyp and f"C{hyp.key}" in self.graph.nodes:
                    self.graph.add_edge(f"C{hyp.key}", claim.id, "supports", "paper.claims")
            for pid in claim.paper_ids:
                self.graph.add_edge(pid, claim.id, "supports", "paper.claims")
            for section in claim.sections:
                statement_id = f"S-{section.strip().lower().replace(' ', '_')}"
                self.graph.add_node(statement_id, "statement", f"Manuscript section: {section}", status="active")
                self.graph.add_edge(claim.id, statement_id, "appears_in", "paper.drafting")
            kept.append(claim)
        result.claims = kept
        self.refresh_claim_states(result)
        self.save_graph()
        return result

    def refresh_claim_states(self, claim_map: ClaimMap) -> ClaimMap:
        for claim in claim_map.claims:
            state, detail = self.graph.derive_state(claim.id, self.settings.allow_synthetic_evidence)
            claim.state = state
            parts = [f"{detail.get('experimental', 0):g} experimental", f"{detail.get('literature', 0):g} literature"]
            if detail.get("refuting"):
                parts.append(f"{detail['refuting']:g} refuting")
            if detail.get("ignored_synthetic"):
                parts.append(f"{len(detail['ignored_synthetic'])} synthetic ignored")
            if detail.get("reason"):
                parts.append(detail["reason"])
            claim.evidence_note = ", ".join(parts)
        self.store.save_claim_map(claim_map)
        return claim_map

    # ------------------------------------------------------------ decisions and roadmap

    def log_decision(
        self,
        kind: str,
        decision: str,
        reason: str = "",
        evidence: List[str] | None = None,
        alternatives: List[str] | None = None,
        agent_inputs: List[str] | None = None,
        effects: List[str] | None = None,
    ) -> Decision:
        item = Decision(
            id=self.store.next_id("D", width=3),
            kind=kind,
            decision=decision,
            reason=reason,
            evidence=list(evidence or []),
            alternatives_considered=list(alternatives or []),
            agent_inputs=list(agent_inputs or []),
            downstream_effects=list(effects or []),
        )
        items = self.store.decisions()
        items.append(item)
        self.store.save_decisions(items)
        return item

    def roadmap(self) -> List[Action]:
        return self.store.roadmap()

    def save_roadmap(self, actions: List[Action]) -> None:
        self.store.save_roadmap(actions)
