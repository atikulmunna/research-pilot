"""Deterministic offline responses for every task kind.

The mock provider lets the full swarm run end to end in tests, CI and demos without API
keys. Outputs are shaped from the structured context each agent passes to the router and
are clearly marked "[mock]". They exercise every loop of the orchestrator (proposal
revision, hypothesis refinement, design revision, multiple review rounds).
"""

import json
from typing import Any, Callable, Dict, List

MOCK_SCRIPT = '''import argparse
import json
import random

ARMS = {arms}
METRICS = {metrics}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-dir", default="artifacts")
args = parser.parse_args()

for index, arm in enumerate(ARMS):
    rng = random.Random(f"{{arm}}-{{args.seed}}")
    metrics = {{}}
    for name, higher_is_better in METRICS:
        value = 0.70 + (0.04 if index == 0 else 0.0) + rng.gauss(0, 0.01)
        metrics[name] = round(value if higher_is_better else 1.0 - value, 6)
    print("RESULT_JSON: " + json.dumps({{"arm": arm, "seed": args.seed, "metrics": metrics}}))
'''


def _topic(context: Dict[str, Any]) -> str:
    return str(context.get("topic") or "the topic")[:80]


class MockResponder:
    def respond(self, task: str, context: Dict[str, Any]) -> Any:
        handler: Callable[[Dict[str, Any]], Any] | None = getattr(self, "_" + task.replace(".", "_"), None)
        return handler(context) if handler else {}

    def _field_search_queries(self, c):
        t = _topic(c)
        return {"queries": [t, f"{t} benchmark", f"{t} evaluation"], "synonyms": [f"{t} methods"]}

    def _field_map(self, c):
        t = _topic(c)
        return {
            "domains": [f"[mock] {t}"],
            "subdomains": ["methods", "evaluation", "robustness"],
            "terminology": [{"term": t, "definition": f"[mock] working definition of {t}", "synonyms": []}],
            "major_problems": ["[mock] generalization under distribution shift"],
            "common_methods": ["[mock] supervised fine-tuning", "[mock] retrieval augmentation"],
            "datasets": ["BenchA", "BenchB"],
            "benchmarks": ["BenchA"],
            "metrics": ["accuracy"],
            "emerging_directions": ["[mock] contrastive objectives"],
            "active_areas": ["[mock] scaling"],
            "neglected_areas": ["[mock] robustness evaluation"],
            "open_questions": ["[mock] does the gain survive distribution shift?"],
            "literature_queries": [f"{t} robustness", f"{t} baselines"],
        }

    def _literature_search_queries(self, c):
        focus = str(c.get("focus") or "")[:60]
        t = _topic(c)
        return {"queries": [f"{t} methods", f"{t} limitations"] + ([focus] if focus else [])}

    def _literature_extraction(self, c):
        out = []
        for i, paper in enumerate(c.get("papers", [])):
            out.append(
                {
                    "id": paper["id"],
                    "problem": f"[mock] problem studied in '{paper['title'][:60]}'",
                    "method": f"[mock] {paper['title'].split(' approach')[0][:40]} method",
                    "novelty": "[mock] combines two known components",
                    "datasets": ["BenchA"],
                    "baselines": ["standard baseline"],
                    "metrics": ["accuracy"],
                    "results": "[mock] reports gains over baselines on BenchA",
                    "limitations": ["degrades under distribution shift", "assumes clean labels"],
                    "assumptions": ["clean labels"],
                    "claims": [{"text": "[mock] the method improves accuracy", "basis": "demonstrated" if i % 2 == 0 else "claimed", "evidence": "BenchA experiments"}],
                    "summary": "[mock] two sentence summary.",
                    "relevance_to_proposal": "[mock] closely related",
                    "relevance_score": 0.5 + (i % 5) / 10,
                    "evidence_quality": ("high", "medium", "low")[i % 3],
                }
            )
        return {"papers": out}

    def _literature_clustering(self, c):
        groups: Dict[str, List[str]] = {}
        for paper in c.get("papers", []):
            name = paper["title"].split(" approach")[0].replace("A ", "").strip()[:30] or "other"
            groups.setdefault(name, []).append(paper["id"])
        ids = [p["id"] for p in c.get("papers", [])]
        return {
            "clusters": [{"name": k, "approach": f"[mock] {k}", "paper_ids": v} for k, v in groups.items()],
            "state_of_the_art": ids[:2],
            "redundant_approaches": [],
            "contradictions": [f"[mock] {ids[0]} and {ids[1]} disagree on robustness"] if len(ids) > 1 else [],
        }

    def _literature_claim_verification(self, c):
        return {"checks": []}

    def _gap_discovery(self, c):
        ids = [p["id"] for p in c.get("papers", [])]
        return {
            "gaps": [
                {
                    "description": "[mock] Robustness under distribution shift is rarely evaluated",
                    "category": "generalization",
                    "supporting_papers": ids[:2],
                    "evidence": "[mock] limitations sections mention degradation under shift",
                    "existing_attempts": "[mock] small-scale robustness checks",
                    "why_unsolved": "[mock] no shared shifted benchmark",
                    "proposed_solution": "[mock] evaluate under controlled shift",
                    "novelty_type": "new evaluation",
                    "novelty_strength": "moderate",
                    "risks": ["[mock] shift may be artificial"],
                },
                {
                    "description": "[mock] Clean-label assumption is untested",
                    "category": "weak_assumption",
                    "supporting_papers": ids[2:4],
                    "novelty_strength": "weak",
                },
            ],
            "coverage_assessment": "[mock] core clusters covered",
            "coverage_sufficient": True,
            "missing_literature_queries": [],
        }

    def _novelty_analysis(self, c):
        ids = list(c.get("papers", []))
        return {
            "closest_prior_work": [{"paper_id": ids[0], "overlap": "[mock] same task", "difference": "[mock] no shift evaluation"}] if ids else [],
            "differentiation": "[mock] adds a controlled distribution-shift evaluation",
            "novelty_type": "new evaluation",
            "novelty_strength": "moderate",
            "verdict": "incremental",
            "confidence": 0.5,
            "why_not_already_covered": "[mock] prior work evaluates in-distribution only",
        }

    def _critique_proposal(self, c):
        first = int(c.get("revision", 0)) == 0
        return {
            "strengths": ["[mock] clear problem statement"],
            "issues": [
                {
                    "category": "novelty_risk",
                    "severity": "major" if first else "minor",
                    "description": "[mock] contribution may be incremental over the closest prior work",
                    "proposed_test": "[mock] compare against the closest prior work under shift",
                },
                {"category": "evaluation_risk", "severity": "minor", "description": "[mock] single benchmark", "proposed_test": "[mock] add a second benchmark"},
            ],
            "required_evidence": ["[mock] shift evaluation with at least three seeds"],
            "verdict": "revise" if first else "proceed",
            "summary": "[mock] defensible after narrowing" if not first else "[mock] needs narrowing",
        }

    def _critique_design(self, c):
        ids = list(c.get("ids", []))
        target = c.get("target", "experiments")
        issues = [
            {
                "category": "baseline_gap" if target == "experiments" else "hidden_assumption",
                "severity": "major",
                "target_id": ids[0] if ids else "",
                "description": f"[mock] {'missing a strong baseline' if target == 'experiments' else 'falsification criterion is vague'}",
                "proposed_test": "[mock] add a strong baseline arm" if target == "experiments" else "[mock] state a numeric threshold",
            }
        ]
        return {"strengths": ["[mock] testable"], "issues": issues, "verdict": "revise", "summary": "[mock] fixable issues"}

    def _proposal_revision(self, c):
        return {
            "action": "revise",
            "revised_proposal": f"{c.get('proposal', '')}\n\n[mock revision] Narrow the scope to robustness under distribution shift and pre-register the evaluation.",
            "rationale": "[mock] addresses the novelty risk by focusing on the under-evaluated setting",
            "changes": ["[mock] narrowed scope", "[mock] pre-registered evaluation"],
        }

    def _hypothesis_design(self, c):
        gaps = list(c.get("gaps", []))
        papers = list(c.get("papers", []))
        motivated = gaps[:1] + papers[:1]
        mode = c.get("mode", "design")
        if mode == "revise":
            bases = c.get("hypotheses") or [c.get("hypothesis", {})]
            return {
                "hypotheses": [
                    {
                        "id": base.get("id", "H1"),
                        "statement": f"{base.get('statement', '[mock] hypothesis')} (revised: threshold of 1 point)",
                        "falsification_criteria": "[mock] improvement below 1 point or p_adj >= 0.05",
                        "independent_variables": base.get("independent_variables", ["method"]),
                        "dependent_variables": base.get("dependent_variables", ["accuracy"]),
                        "motivated_by": base.get("motivated_by", motivated),
                        "priority": base.get("priority", 1),
                    }
                    for base in bases
                ]
            }
        if mode == "follow_up":
            return {"hypotheses": [{"statement": f"[mock] {c.get('statement', 'follow-up effect')}", "falsification_criteria": "[mock] no effect", "motivated_by": motivated, "priority": 3}]}
        return {
            "question_decomposition": ["[mock] does the method help in distribution?", "[mock] does the gain survive shift?"],
            "hypotheses": [
                {
                    "statement": "[mock] The proposed method improves accuracy over the baseline on BenchA",
                    "independent_variables": ["method"],
                    "dependent_variables": ["accuracy"],
                    "controls": ["same compute budget"],
                    "expected_outcome": "[mock] +3 points",
                    "falsification_criteria": "[mock] no significant improvement",
                    "competing_hypotheses": ["[mock] gain comes from extra compute"],
                    "motivated_by": motivated,
                    "priority": 1,
                },
                {
                    "statement": "[mock] The improvement persists under distribution shift",
                    "independent_variables": ["method", "shift"],
                    "dependent_variables": ["accuracy"],
                    "falsification_criteria": "[mock] gain vanishes under shift",
                    "motivated_by": motivated,
                    "priority": 2,
                },
            ],
        }

    def _experiment_design(self, c):
        mode = c.get("mode", "design")
        kind = c.get("kind", "primary")
        out = []
        sources = c.get("existing") if mode == "revise" else [{"hypothesis_id": h} for h in c.get("hypotheses", [])]
        for item in sources or []:
            baselines = ["baseline", "strong_baseline"] if mode == "revise" else ["baseline"]
            out.append(
                {
                    "id": item.get("id", "") if mode == "revise" else "",
                    "hypothesis_id": item["hypothesis_id"],
                    "kind": item.get("kind", kind) if mode == "revise" else kind,
                    "objective": f"[mock] {kind} test of {item['hypothesis_id']}",
                    "datasets": ["BenchA"],
                    "method": "proposed",
                    "baselines": baselines,
                    "controls": ["same compute"],
                    "metrics": [{"name": "accuracy", "higher_is_better": True, "primary": True, "min_effect": 0.01}],
                    "seeds": [0, 1, 2],
                    "expected_outcomes": "[mock] proposed > baseline",
                    "failure_criteria": "[mock] p_adj >= 0.05 or diff < 0.01",
                    "statistical_plan": "[mock] Welch t-test with Holm correction",
                    "confounders": ["[mock] tuning budget"],
                }
            )
        return {"experiments": out, "notes": "[mock]"}

    def _experiment_coding(self, c):
        spec = c.get("spec", {})
        arms = [spec.get("method") or "proposed", *(spec.get("baselines") or [])]
        metrics = [(m.get("name", "accuracy"), bool(m.get("higher_is_better", True))) for m in spec.get("metrics", [])] or [("accuracy", True)]
        script = MOCK_SCRIPT.format(arms=json.dumps(arms), metrics=repr(metrics))
        notes = {"dataset_version": "mock-bench-v1", "model_version": "mock", "compute_estimate": "seconds on CPU", "assumptions": ["[mock] toy simulation"], "deviations": []}
        return f"```python\n{script}```\n\n```json\n{json.dumps(notes)}\n```"

    def _experiment_debugging(self, c):
        return self._experiment_coding(c)

    def _stats_summary(self, c):
        return {"summary": "[mock] The method arm outperforms the baselines on the primary metric.", "interpretation_constraints": ["[mock] small number of seeds"]}

    def _science_interpretation(self, c):
        run_id = c.get("run_id", "")
        supported = bool(c.get("primary_significant"))
        return {
            "verdict": "supported" if supported else "inconclusive",
            "verdict_rationale": "[mock] primary comparisons are significant in the expected direction" if supported else "[mock] effect not reliable",
            "statements": [
                {"text": "[mock] the method arm scores higher on the primary metric", "tag": "evidence_backed", "evidence_refs": [run_id]},
                {"text": "[mock] the gain may come from better regularisation", "tag": "speculation"},
            ],
            "competing_explanations": [{"explanation": "[mock] benchmark artifact", "plausibility": 0.2, "follow_up": "[mock] evaluate on a held-out benchmark"}],
            "unexpected_findings": [],
            "contradicting_evidence": [],
            "follow_up_hypotheses": [{"statement": "[mock] the gain grows with model size", "rationale": "[mock] trend across arms"}],
        }

    def _research_planning(self, c):
        hints = list(c.get("hints", []))[:3]
        return {
            "assessment": "[mock] follow the state-derived candidates",
            "evidence_sufficient_for_paper": bool(c.get("evidence_sufficient")),
            "candidates": hints,
            "abandon": [],
            "blockers": [],
        }

    def _paper_claims(self, c):
        claims = []
        for idx, hyp in enumerate(c.get("hypotheses", [])):
            claims.append(
                {
                    "text": hyp.get("statement", ""),
                    "kind": "contribution",
                    "importance": "central" if idx == 0 else "supporting",
                    "hypothesis_ids": [hyp["id"]],
                    "run_ids": c.get("runs", {}).get(hyp["id"], []),
                    "sections": ["results", "discussion"],
                }
            )
        papers = list(c.get("papers", []))
        if papers:
            claims.append({"text": "[mock] prior work rarely evaluates under shift", "kind": "background", "paper_ids": papers[:2], "sections": ["introduction", "related_work"]})
        claims.append({"text": "[mock] results are limited to one benchmark", "kind": "limitation", "sections": ["limitations"]})
        return {"storyline": "[mock] gap, method, evidence, limits", "claims": claims}

    def _paper_drafting(self, c):
        papers = list(c.get("papers", []))
        cite = f"[{papers[0]}]" if papers else ""
        return {
            "title": f"[mock] {c.get('title', 'Research report')}",
            "abstract": "[mock] We study the problem, propose a method, and test it with pre-registered experiments. Results are reported with uncertainty.",
            "introduction": f"[mock] Prior work {cite} rarely evaluates robustness.",
            "related_work": f"[mock] Closest work {cite}.",
            "method": "[mock] The proposed method.",
            "experimental_setup": "[mock] Three seeds per arm; Welch t-tests with Holm correction.",
            "results": "[mock] See Table 1 for all arms and comparisons.",
            "discussion": "[mock] Competing explanations remain open.",
            "limitations": "[mock] Synthetic data; single benchmark.",
            "conclusion": "[mock] Preliminary evidence only.",
        }

    def _text_rewriting(self, c):
        return "[mock] Plain-language summary of the study."

    def _review_simulation(self, c):
        round_no = int(c.get("round", 1))
        issues = [{"severity": "minor", "dimension": "writing", "description": "[mock] tighten the abstract", "required_action": "revision", "suggestion": "[mock] shorten"}]
        if round_no == 1:
            issues.insert(0, {"severity": "major", "dimension": "experiments", "description": "[mock] no ablation isolates the key component", "required_action": "experiment", "suggestion": "[mock] add an ablation"})
        return {
            "recommendation": "major_revision" if round_no == 1 else "minor_revision",
            "summary": f"[mock] review round {round_no}",
            "issues": issues,
            "missing_experiments": ["[mock] ablation"] if round_no == 1 else [],
            "unsupported_claims": [],
            "likely_reviewer_questions": ["[mock] how sensitive is the result to seeds?"],
            "resolved_issue_ids": list(c.get("open_issues", [])) if round_no > 1 else [],
        }
