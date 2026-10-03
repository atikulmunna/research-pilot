import re
from typing import Dict, List, Tuple

from ..services.bibliography import bibliography
from ..state.models import MANUSCRIPT_SECTIONS, ClaimMap, Draft, ExperimentSpec, TaskEnvelope
from .base import Agent, clip, paper_digest, ranked_papers
from .quant_analyst import comparison_table, summary_table

CITE = re.compile(r"\bP\d{3,}\b")
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\w.])")
SECTION_TITLES = {
    "introduction": "Introduction",
    "related_work": "Related Work",
    "method": "Method",
    "experimental_setup": "Experimental Setup",
    "results": "Results",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
}


class PaperArchitect(Agent):
    key = "paper_architect"
    name = "Paper Architect"

    def run(self, task: TaskEnvelope):
        revising = bool(self.store.manuscript())
        evidence = self._evidence_block()
        papers = ranked_papers(p for p in self.state.papers() if p.extracted)[:20]
        hyps = self.state.current_hypotheses()
        runs = [r for r in self.state.experiments.runs() if r.status == "completed"]

        claim_map = self.ask(
            "paper.claims",
            f"{self.project_block()}\n\n{evidence}\n\nLiterature:\n{paper_digest(papers, detail=False)}\n\n"
            "Define the paper's contribution claims and map each to its evidence (hypothesis ids, run ids, paper ids). "
            "Mark at most three claims as central. Include background claims for the related work and limitation claims.",
            ClaimMap,
            {
                "hypotheses": [{"id": h.id, "statement": h.statement, "status": h.status} for h in hyps],
                "runs": {h.id: [r.id for r in runs if r.hypothesis_id == h.id] for h in hyps},
                "papers": [p.id for p in papers],
            },
        )
        claim_map = self.state.set_manuscript_claims(claim_map)

        tables = self._tables()
        for idx, (caption, body) in enumerate(tables, start=1):
            self.store.write_text(f"paper/tables/table_{idx}.md", f"**{caption}**\n\n{body}\n")

        claims_text = "\n".join(
            f"- {c.id} [{c.kind}, {c.importance}, evidence state {c.state}] {c.text} "
            f"(hypotheses {c.hypothesis_ids or '-'}, runs {c.run_ids or '-'}, papers {c.paper_ids or '-'})"
            for c in claim_map.claims
        )
        table_text = "\n\n".join(f"{caption}\n{body}" for caption, body in tables) or "(no completed experiments)"
        parts = [
            self.project_block(),
            f"Claim-evidence map:\n{claims_text}",
            f"Result tables (the only source of numbers you may use; refer to them as Table N):\n{table_text}",
            evidence,
            f"Papers you may cite as [P###]:\n{paper_digest(papers, detail=False)}",
            "Writing rules: claims whose evidence state is not SUPPORTED must be phrased as preliminary or hypothetical. "
            "Report negative and inconclusive results. Put synthetic, missing or contradictory evidence in Limitations.",
        ]
        issues = [i for r in self.store.reviews() for i in r.issues if i.status == "open"]
        if revising:
            previous = self.store.draft()
            if previous:
                parts.append(f"Previous draft abstract:\n{clip(previous.abstract, 1200)}")
            if issues:
                parts.append("Address these open review issues where writing can address them:\n" + "\n".join(f"- {i.id} [{i.severity}, {i.dimension}] {i.description} -> {i.suggestion}" for i in issues))
        draft = self.ask(
            "paper.drafting",
            "\n\n".join(parts) + "\n\nWrite the full paper as markdown text per section (no section headings inside the fields).",
            Draft,
            {"claims": [c.model_dump() for c in claim_map.claims], "papers": [p.id for p in papers], "title": self.project().title},
        )
        draft, qa = self._quality_checks(draft, claim_map, tables)
        lay = ""
        if draft.abstract:
            lay = self.ask_text(
                "text.rewriting",
                f"Rewrite this abstract as a plain-language summary of at most three sentences. Do not add any claim.\n\n{draft.abstract}",
                {"text": draft.abstract},
            ).strip()
        self.store.save_draft(draft)
        self.store.write_yaml("paper/qa.yaml", {**qa, "lay_summary": lay})
        self.store.write_text("paper/supplementary/reproducibility.md", self._reproducibility())
        self.render_manuscript()
        return self.result(
            task,
            findings=[
                f"{'Revised' if revising else 'Drafted'} manuscript with {len(claim_map.claims)} claims and {len(tables)} result tables",
                "Claim states: " + ", ".join(f"{c.id}={c.state}" for c in claim_map.claims),
            ],
            uncertainties=[f"{k}: {v}" for k, v in qa.items() if v and k != "unsupported_statements"],
            artifacts=["paper/manuscript/manuscript.md", "paper/claims.yaml", "paper/qa.yaml", "paper/supplementary/reproducibility.md"],
            data={"claims": len(claim_map.claims), "qa": {k: len(v) for k, v in qa.items()}},
        )

    # ------------------------------------------------------------ evidence and tables (code)

    def _evidence_block(self) -> str:
        registry = self.state.experiments
        lines = ["Evidence status:"]
        for hyp in self.state.current_hypotheses():
            lines.append(f"- {hyp.key} [{hyp.status}, evidence {self.state.claim_state(f'C{hyp.key}')}] {clip(hyp.statement, 200)}")
            for spec in [s for s in registry.latest_specs() if s.hypothesis_id == hyp.id]:
                lines.extend(self._run_lines(spec))
        superseded = [h for h in self.state.hypotheses() if h.status == "superseded"]
        if superseded:
            lines.append("Revised hypotheses (must be reported transparently):")
            lines.extend(f"- {h.key} -> superseded" for h in superseded)
        return "\n".join(lines)

    def _run_lines(self, spec: ExperimentSpec) -> List[str]:
        lines = []
        for run in self.state.experiments.runs_for(spec.key):
            interp = self.store.interpretation(run.id)
            verdict = interp.verdict if interp else run.status
            lines.append(f"    - {run.id} ({spec.key}, {spec.kind}): {verdict}" + (" [synthetic]" if run.synthetic else ""))
            if interp:
                lines.append(f"      interpretation: {clip(interp.verdict_rationale, 300)}")
        return lines

    def _tables(self) -> List[Tuple[str, str]]:
        registry = self.state.experiments
        tables = []
        for run in registry.runs():
            quant = self.store.quant_result(run.id)
            if run.status != "completed" or quant is None:
                continue
            spec = registry.spec(run.spec_key)
            caption = f"Table {len(tables) + 1}: {run.id}, {spec.key} ({spec.kind}) testing {quant.hypothesis_id}, seeds {run.seeds}"
            if run.synthetic:
                caption += " [synthetic, not evidence]"
            tables.append((caption, f"{summary_table(quant)}\n\n{comparison_table(quant)}"))
        return tables

    def _quality_checks(self, draft: Draft, claim_map: ClaimMap, tables: List[Tuple[str, str]]) -> Tuple[Draft, Dict[str, List]]:
        known = self.state.paper_ids()
        unknown: List[str] = []

        def scrub(text: str) -> str:
            def repl(match: re.Match) -> str:
                if match.group(0) in known:
                    return match.group(0)
                unknown.append(match.group(0))
                return "UNVERIFIED-CITATION"

            return CITE.sub(repl, text or "")

        for field in ("abstract", *MANUSCRIPT_SECTIONS):
            setattr(draft, field, scrub(getattr(draft, field)))

        allowed = set(DECIMAL.findall("\n".join(body for _, body in tables)))
        for quant in self.store.quant_results():
            values = [v for c in quant.comparisons for v in (c.diff, c.p_value, c.p_adjusted, c.hedges_g, c.ci_low, c.ci_high, c.rel_diff) if v is not None]
            values += [v for s in quant.summaries for v in (s.mean, s.std, s.ci_low, s.ci_high) if v is not None]
            for value in values:
                for digits in range(1, 5):
                    allowed.add(f"{abs(value):.{digits}f}")
                    allowed.add(f"{abs(value) * 100:.{max(0, digits - 2)}f}")
        narrative = " ".join([draft.abstract, draft.results, draft.discussion, draft.conclusion])
        unverified = sorted({n for n in DECIMAL.findall(narrative) if n not in allowed})

        cited = set(CITE.findall(" ".join(getattr(draft, f) for f in ("abstract", *MANUSCRIPT_SECTIONS))))
        papers = {p.id: p for p in self.state.papers()}
        qa = {
            "unknown_citations": sorted(set(unknown)),
            "synthetic_citations": sorted(pid for pid in cited if pid in papers and papers[pid].synthetic),
            "unverified_numbers": unverified[:25],
            "claims_exceeding_evidence": [
                c.id for c in claim_map.claims if c.kind in {"contribution", "finding"} and c.state not in {"SUPPORTED", "PARTIALLY_SUPPORTED"}
            ],
            "unsupported_statements": [s["statement"] for s in self.state.graph.unsupported_statements(self.settings.allow_synthetic_evidence)],
        }
        for pid in qa["unknown_citations"]:
            self.state.violation("no_fabricated_evidence", f"manuscript cited unknown paper {pid}; replaced with UNVERIFIED-CITATION")
        return draft, qa

    def _reproducibility(self) -> str:
        registry = self.state.experiments
        lines = ["# Reproducibility record", "", "Every run is listed, including failed, inconclusive and synthetic ones.", ""]
        lines.append("| run | experiment | status | code version | seeds | executor | synthetic | attempts | deviations |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for run in registry.runs():
            deviations = "; ".join(run.deviations) or "none"
            lines.append(
                f"| {run.id} | {run.spec_key} | {run.status} | {run.code_version} | {run.seeds} | {run.executor} | "
                f"{'yes' if run.synthetic else 'no'} | {len(run.attempts)} | {clip(deviations, 200)} |"
            )
        lines.append("")
        for spec in registry.specs():
            if len(spec.change_log) > 1:
                lines.append(f"- {spec.key} change log: " + " | ".join(spec.change_log))
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------ assembly (code)

    def render_manuscript(self, status: str = "Draft (pre-review)") -> str:
        draft = self.store.draft()
        claim_map = self.store.claim_map()
        if draft is None or claim_map is None:
            return ""
        claim_map = self.state.refresh_claim_states(claim_map)
        qa = self.store.read_yaml("paper/qa.yaml", {}) or {}
        project = self.project()
        central = [c for c in claim_map.claims if c.importance == "central"]
        supported = [c for c in central if c.state == "SUPPORTED"]
        synthetic = any(r.synthetic for r in self.state.experiments.runs()) or any(p.synthetic for p in self.state.papers())
        lines = [f"# {draft.title or project.title}", ""]
        lines.append(
            f"> **Status:** {status}. {len(supported)} of {len(central)} central claims are SUPPORTED by the evidence graph "
            f"({len(self.store.reviews())} red-team review round(s))."
        )
        if synthetic:
            lines.append("> **Warning:** some sources or results are synthetic (mock or simulated). They demonstrate the pipeline and are not scientific evidence.")
        if qa.get("unknown_citations"):
            lines.append(f"> {len(qa['unknown_citations'])} unverifiable citation(s) were replaced with UNVERIFIED-CITATION.")
        if qa.get("unverified_numbers"):
            lines.append(f"> Numbers not traceable to the result tables: {', '.join(qa['unverified_numbers'][:10])}.")
        lines.append("")
        if qa.get("lay_summary"):
            lines.extend([f"**Plain-language summary.** {qa['lay_summary']}", ""])
        lines.extend(["## Abstract", "", draft.abstract, ""])
        tables = sorted(self.store.path("paper/tables").glob("table_*.md"), key=lambda p: int(p.stem.split("_")[1]))
        for idx, field in enumerate(MANUSCRIPT_SECTIONS, start=1):
            lines.extend([f"## {idx}. {SECTION_TITLES[field]}", "", getattr(draft, field) or "_(not written)_", ""])
            if field == "results" and tables:
                lines.extend(["### Result tables", ""])
                for path in tables:
                    lines.extend([path.read_text(encoding="utf-8").strip(), ""])
        lines.extend(["## Claim-Evidence Map", "", "| claim | kind | importance | evidence state | traceability |", "|---|---|---|---|---|"])
        for claim in claim_map.claims:
            trace = ", ".join(claim.hypothesis_ids + claim.run_ids + claim.paper_ids) or "none"
            lines.append(f"| {claim.id}: {clip(claim.text, 160)} | {claim.kind} | {claim.importance} | {claim.state} | {trace}; {claim.evidence_note} |")
        lines.append("")
        cited = set(CITE.findall(" ".join(getattr(draft, f) for f in ("abstract", *MANUSCRIPT_SECTIONS))))
        refs = [p for p in self.state.papers() if p.id in cited]
        lines.extend(["## References", ""])
        lines.extend(bibliography(refs) or ["_(no papers cited)_"])
        markdown = "\n".join(lines).rstrip() + "\n"
        self.store.save_manuscript(markdown)
        return markdown
