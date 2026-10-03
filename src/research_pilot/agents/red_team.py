from typing import List, Set

from ..state.models import ClaimMap, ExperimentRun, ReviewIssue, ReviewReport, TaskEnvelope, now_iso
from .base import Agent, as_yaml, clip


def _run_line(run: ExperimentRun) -> str:
    deviations = " deviations: " + "; ".join(run.deviations) if run.deviations else ""
    return f"- {run.id} {run.spec_key} {run.status}{' synthetic' if run.synthetic else ''}{deviations}"


class RedTeamReviewer(Agent):
    key = "red_team"
    name = "Red-Team Reviewer"

    def run(self, task: TaskEnvelope):
        reports = self.store.reviews()
        round_no = len(reports) + 1
        open_issues = [i for r in reports for i in r.issues if i.status == "open"]
        claim_map = self.store.claim_map()
        report = self.ask(
            "review.simulation",
            self._prompt(claim_map, open_issues),
            ReviewReport,
            {"round": round_no, "open_issues": [i.id for i in open_issues], "claims": [{"id": c.id, "state": c.state} for c in (claim_map.claims if claim_map else [])]},
        )
        report.round = round_no
        report.created_at = now_iso()
        self._number_issues(report, claim_map)
        resolved = set(report.resolved_issue_ids) & {i.id for i in open_issues}
        report.resolved_issue_ids = sorted(resolved)
        self._mark_resolved(reports, resolved)
        self.store.save_review(report)
        still_open = [i for i in open_issues if i.id not in resolved] + report.issues
        return self._result(task, report, len(resolved), still_open)

    def _prompt(self, claim_map: ClaimMap | None, open_issues: List[ReviewIssue]) -> str:
        qa = self.store.read_yaml("paper/qa.yaml", {}) or {}
        novelty = self.state.latest_novelty()
        runs = self.state.experiments.runs()
        parts = [
            "Manuscript under review:\n" + clip(self.store.manuscript(), 16000),
            "Claim states derived from the evidence graph:\n"
            + ("\n".join(f"- {c.id} [{c.kind}, {c.importance}] {c.state}: {clip(c.text, 160)}" for c in claim_map.claims) if claim_map else "(none)"),
            "Automated quality checks:\n" + as_yaml({k: v for k, v in qa.items() if k != "lay_summary"}),
            "All experiment runs (including failures):\n" + "\n".join(_run_line(r) for r in runs),
        ]
        if novelty:
            parts.append(f"Novelty analysis: {novelty.verdict} ({novelty.novelty_strength}), searched {novelty.search_record.papers_checked if novelty.search_record else '?'} papers")
        if open_issues:
            parts.append("Issues raised in earlier rounds (decide which are now resolved):\n" + "\n".join(f"- {i.id} [{i.severity}, {i.dimension}] {i.description}" for i in open_issues))
        parts.append("Review this paper as a skeptical reviewer at a top venue. Find every reason it could be rejected.")
        return "\n\n".join(parts)

    def _number_issues(self, report: ReviewReport, claim_map: ClaimMap | None) -> None:
        claim_ids = {c.id for c in claim_map.claims} if claim_map else set()
        for idx, issue in enumerate(report.issues, start=1):
            issue.id = f"RT{report.round}-{idx}"
            issue.round = report.round
            issue.affected_claims = self.state.filter_refs(issue.affected_claims, claim_ids, f"review issue {issue.id}")

    def _mark_resolved(self, reports: List[ReviewReport], resolved: Set[str]) -> None:
        for previous in reports:
            hits = [issue for issue in previous.issues if issue.id in resolved]
            for issue in hits:
                issue.status = "resolved"
            if hits:
                self.store.save_review(previous)

    def _result(self, task: TaskEnvelope, report: ReviewReport, resolved_count: int, still_open: List[ReviewIssue]):
        counts = {s: sum(1 for i in report.issues if i.severity == s) for s in ("critical", "major", "minor")}
        return self.result(
            task,
            findings=[
                f"Round {report.round}: {report.recommendation}; {counts['critical']} critical, {counts['major']} major, {counts['minor']} minor; "
                f"{resolved_count} earlier issue(s) resolved"
            ],
            recommendations=[f"{i.id}: {i.suggestion}" for i in report.issues if i.severity != "minor" and i.suggestion],
            uncertainties=list(report.likely_reviewer_questions[:5]),
            artifacts=[f"reviews/red_team/review_{report.round:02d}.yaml"],
            data={
                "round": report.round,
                "recommendation": report.recommendation,
                "open_serious": sum(1 for i in still_open if i.severity in {"critical", "major"}),
            },
        )
