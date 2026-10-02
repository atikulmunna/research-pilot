from ..state.models import ReviewReport, TaskEnvelope, now_iso
from .base import Agent, as_yaml, clip


class RedTeamReviewer(Agent):
    key = "red_team"
    name = "Red-Team Reviewer"

    def run(self, task: TaskEnvelope):
        reports = self.store.reviews()
        round_no = len(reports) + 1
        open_issues = [i for r in reports for i in r.issues if i.status == "open"]
        claim_map = self.store.claim_map()
        qa = self.store.read_yaml("paper/qa.yaml", {}) or {}
        novelty = self.state.latest_novelty()
        runs = self.state.experiments.runs()
        parts = [
            "Manuscript under review:\n" + clip(self.store.manuscript(), 16000),
            "Claim states derived from the evidence graph:\n"
            + ("\n".join(f"- {c.id} [{c.kind}, {c.importance}] {c.state}: {clip(c.text, 160)}" for c in claim_map.claims) if claim_map else "(none)"),
            "Automated quality checks:\n" + as_yaml({k: v for k, v in qa.items() if k != "lay_summary"}),
            "All experiment runs (including failures):\n"
            + "\n".join(f"- {r.id} {r.spec_key} {r.status}{' synthetic' if r.synthetic else ''}{' deviations: ' + '; '.join(r.deviations) if r.deviations else ''}" for r in runs),
        ]
        if novelty:
            parts.append(f"Novelty analysis: {novelty.verdict} ({novelty.novelty_strength}), searched {novelty.search_record.papers_checked if novelty.search_record else '?'} papers")
        if open_issues:
            parts.append("Issues raised in earlier rounds (decide which are now resolved):\n" + "\n".join(f"- {i.id} [{i.severity}, {i.dimension}] {i.description}" for i in open_issues))
        parts.append("Review this paper as a skeptical reviewer at a top venue. Find every reason it could be rejected.")
        report = self.ask(
            "review.simulation",
            "\n\n".join(parts),
            ReviewReport,
            {"round": round_no, "open_issues": [i.id for i in open_issues], "claims": [{"id": c.id, "state": c.state} for c in (claim_map.claims if claim_map else [])]},
        )
        report.round = round_no
        report.created_at = now_iso()
        claim_ids = {c.id for c in claim_map.claims} if claim_map else set()
        for idx, issue in enumerate(report.issues, start=1):
            issue.id = f"RT{round_no}-{idx}"
            issue.round = round_no
            issue.affected_claims = self.state.filter_refs(issue.affected_claims, claim_ids, f"review issue {issue.id}")
        resolved = set(report.resolved_issue_ids) & {i.id for i in open_issues}
        report.resolved_issue_ids = sorted(resolved)
        for previous in reports:
            changed = False
            for issue in previous.issues:
                if issue.id in resolved:
                    issue.status = "resolved"
                    changed = True
            if changed:
                self.store.save_review(previous)
        self.store.save_review(report)
        counts = {s: sum(1 for i in report.issues if i.severity == s) for s in ("critical", "major", "minor")}
        still_open = [i for i in open_issues if i.id not in resolved] + report.issues
        return self.result(
            task,
            findings=[
                f"Round {round_no}: {report.recommendation}; {counts['critical']} critical, {counts['major']} major, {counts['minor']} minor; "
                f"{len(resolved)} earlier issue(s) resolved"
            ],
            recommendations=[f"{i.id}: {i.suggestion}" for i in report.issues if i.severity != "minor" and i.suggestion],
            uncertainties=list(report.likely_reviewer_questions[:5]),
            artifacts=[f"reviews/red_team/review_{round_no:02d}.yaml"],
            data={
                "round": round_no,
                "recommendation": report.recommendation,
                "open_serious": sum(1 for i in still_open if i.severity in {"critical", "major"}),
            },
        )
