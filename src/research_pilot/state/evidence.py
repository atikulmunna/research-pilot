"""Evidence graph: papers, claims, hypotheses, experiments, results and paper statements.

Claim states are derived from the graph, never accepted from an agent. An agent can
propose that a claim is supported, but the state only becomes SUPPORTED when the graph
holds enough non-synthetic supporting evidence.
"""

from typing import Any, Dict, Iterable, List, Tuple

from pydantic import Field

from .models import Model

NODE_TYPES = {"paper", "claim", "hypothesis", "experiment", "result", "gap", "statement"}
RELATIONS = {
    "supports",
    "contradicts",
    "extends",
    "cites",
    "motivates",
    "tested_by",
    "produces",
    "refutes",
    "suggests",
    "appears_in",
}
EXPERIMENTAL_KINDS = {"contribution", "finding"}
INACTIVE = {"superseded", "abandoned"}
UNCERTAINTY = {
    "UNKNOWN": 1.0,
    "HYPOTHESIS": 1.0,
    "SPECULATION": 0.9,
    "PARTIALLY_SUPPORTED": 0.6,
    "CONTRADICTED": 0.4,
    "SUPPORTED": 0.1,
}


class GraphNode(Model):
    id: str
    type: str
    label: str = ""
    attrs: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(Model):
    source: str
    target: str
    relation: str
    provenance: str = ""
    attrs: Dict[str, Any] = Field(default_factory=dict)


class EvidenceGraph:
    def __init__(self, nodes: Iterable[GraphNode] = (), edges: Iterable[GraphEdge] = ()):
        self.nodes: Dict[str, GraphNode] = {node.id: node for node in nodes}
        self.edges: List[GraphEdge] = list(edges)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceGraph":
        return cls(
            nodes=[GraphNode.model_validate(n) for n in data.get("nodes", [])],
            edges=[GraphEdge.model_validate(e) for e in data.get("edges", [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump(mode="json") for n in self.nodes.values()],
            "edges": [e.model_dump(mode="json") for e in self.edges],
        }

    # ------------------------------------------------------------ mutation

    def add_node(self, node_id: str, node_type: str, label: str = "", **attrs: Any) -> GraphNode:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type}")
        node = self.nodes.get(node_id)
        if node is None:
            node = GraphNode(id=node_id, type=node_type, label=label, attrs=attrs)
            self.nodes[node_id] = node
        else:
            if label:
                node.label = label
            node.attrs.update(attrs)
        return node

    def add_edge(self, source: str, target: str, relation: str, provenance: str = "", **attrs: Any) -> GraphEdge:
        if relation not in RELATIONS:
            raise ValueError(f"Unknown relation: {relation}")
        missing = [n for n in (source, target) if n not in self.nodes]
        if missing:
            raise ValueError(f"Edge {source} -{relation}-> {target} references unknown node(s): {missing}")
        for edge in self.edges:
            if edge.source == source and edge.target == target and edge.relation == relation:
                edge.attrs.update(attrs)
                return edge
        edge = GraphEdge(source=source, target=target, relation=relation, provenance=provenance, attrs=attrs)
        self.edges.append(edge)
        return edge

    # ------------------------------------------------------------ traversal

    def node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(node_id)

    def nodes_of(self, node_type: str, include_inactive: bool = False) -> List[GraphNode]:
        return [
            n
            for n in self.nodes.values()
            if n.type == node_type and (include_inactive or n.attrs.get("status") not in INACTIVE)
        ]

    def incoming(self, node_id: str, relations: Iterable[str] | None = None) -> List[GraphEdge]:
        rel = set(relations) if relations else None
        return [e for e in self.edges if e.target == node_id and (rel is None or e.relation in rel)]

    def outgoing(self, node_id: str, relations: Iterable[str] | None = None) -> List[GraphEdge]:
        rel = set(relations) if relations else None
        return [e for e in self.edges if e.source == node_id and (rel is None or e.relation in rel)]

    # ------------------------------------------------------------ claim states

    def derive_state(self, claim_id: str, allow_synthetic: bool = False, _seen: set | None = None) -> Tuple[str, Dict[str, Any]]:
        node = self.nodes.get(claim_id)
        if node is None or node.type != "claim":
            return "UNKNOWN", {"reason": "not a claim"}
        seen = set(_seen or ())
        if claim_id in seen:
            return "UNKNOWN", {"reason": "cycle"}
        seen.add(claim_id)

        tally: Dict[str, Any] = {"experimental": 0.0, "literature": 0.0, "refuting": 0.0, "sources": [], "ignored_synthetic": [], "untested_source": False}
        for edge in self.incoming(claim_id, {"supports", "refutes", "contradicts"}):
            src = self.nodes.get(edge.source)
            if src is not None:
                self._count_edge(tally, edge, src, allow_synthetic, seen)
        untested_source = tally.pop("untested_source")
        support = tally["experimental"] + tally["literature"]
        detail = {**tally, "contested": bool(tally["refuting"] and support)}
        return self._classify(claim_id, node, detail, support, untested_source), detail

    def _count_edge(self, tally: Dict[str, Any], edge: GraphEdge, src: GraphNode, allow_synthetic: bool, seen: set) -> None:
        if (src.attrs.get("synthetic") or edge.attrs.get("synthetic")) and not allow_synthetic:
            tally["ignored_synthetic"].append(src.id)
            return
        tally["sources"].append(src.id)
        if edge.relation != "supports":
            tally["refuting"] += 1
        elif src.type == "result":
            tally["experimental"] += 1
        elif src.type == "paper":
            weak = edge.attrs.get("basis") == "claimed" or src.attrs.get("evidence_quality") == "low"
            tally["literature"] += 0.5 if weak else 1.0
        elif src.type == "claim":
            self._count_claim_source(tally, src.id, allow_synthetic, seen)

    def _count_claim_source(self, tally: Dict[str, Any], source_id: str, allow_synthetic: bool, seen: set) -> None:
        """A claim supported by another claim inherits that claim's evidence, by kind."""
        sub_state, sub = self.derive_state(source_id, allow_synthetic, seen)
        tally["untested_source"] = tally["untested_source"] or sub_state == "HYPOTHESIS"
        if sub_state == "CONTRADICTED":
            tally["refuting"] += 1
            return
        weight = {"SUPPORTED": 2.0, "PARTIALLY_SUPPORTED": 1.0}.get(sub_state, 0.0)
        tally["experimental" if sub.get("experimental", 0) > 0 else "literature"] += weight

    def _classify(self, claim_id: str, node: GraphNode, detail: Dict[str, Any], support: float, untested_source: bool) -> str:
        if detail["refuting"]:
            return "PARTIALLY_SUPPORTED" if support else "CONTRADICTED"
        if support == 0:
            if untested_source or node.attrs.get("hypothesis_claim") or self.outgoing(claim_id, {"motivates"}):
                return "HYPOTHESIS"
            return "SPECULATION" if node.attrs.get("speculative") else "UNKNOWN"
        if node.attrs.get("kind", "finding") in EXPERIMENTAL_KINDS and detail["experimental"] == 0:
            detail["reason"] = "untested: literature can motivate this claim but not establish it"
            return "HYPOTHESIS"
        return "SUPPORTED" if support >= 2 else "PARTIALLY_SUPPORTED"

    def claim_states(self, allow_synthetic: bool = False) -> Dict[str, str]:
        return {n.id: self.derive_state(n.id, allow_synthetic)[0] for n in self.nodes_of("claim")}

    # ------------------------------------------------------------ queries from the plan

    def claims_with_single_support(self, allow_synthetic: bool = False) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes_of("claim"):
            _, detail = self.derive_state(node.id, allow_synthetic)
            if len(detail.get("sources", [])) == 1 and not detail.get("refuting"):
                out.append({"claim": node.id, "label": node.label, "source": detail["sources"][0]})
        return out

    def contradicted_claims(self, allow_synthetic: bool = False) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes_of("claim"):
            state, detail = self.derive_state(node.id, allow_synthetic)
            if state == "CONTRADICTED" or detail.get("contested"):
                out.append({"claim": node.id, "label": node.label, "state": state, "sources": detail["sources"]})
        return out

    def untested_hypotheses(self) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes_of("hypothesis"):
            experiments = [e.target for e in self.outgoing(node.id, {"tested_by"})]
            with_results = [x for x in experiments if self.outgoing(x, {"produces"})]
            if not with_results:
                out.append(
                    {
                        "hypothesis": node.id,
                        "label": node.label,
                        "reason": "not_run" if experiments else "no_experiment",
                        "experiments": experiments,
                    }
                )
        return out

    def experiments_for_claim(self, claim_id: str) -> List[str]:
        hypotheses = [e.target for e in self.outgoing(claim_id, {"motivates"}) if self.nodes[e.target].type == "hypothesis"]
        for edge in self.incoming(claim_id, {"supports", "refutes", "contradicts"}):
            src = self.nodes.get(edge.source)
            if src and src.type == "claim":
                hypotheses.extend(e.target for e in self.outgoing(src.id, {"motivates"}) if self.nodes[e.target].type == "hypothesis")
        experiments: List[str] = []
        for hyp in hypotheses:
            experiments.extend(e.target for e in self.outgoing(hyp, {"tested_by"}))
        return sorted(set(experiments))

    def unsupported_statements(self, allow_synthetic: bool = False) -> List[Dict[str, Any]]:
        out = []
        weak = {"UNKNOWN", "SPECULATION", "HYPOTHESIS", "CONTRADICTED"}
        for node in self.nodes_of("statement"):
            claims = [
                e.source
                for e in self.incoming(node.id, {"appears_in"})
                if self.nodes[e.source].attrs.get("status") not in INACTIVE
            ]
            bad = [c for c in claims if self.derive_state(c, allow_synthetic)[0] in weak]
            if bad:
                out.append({"statement": node.id, "label": node.label, "claims": bad})
        return out

    def weak_literature_claims(self) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes_of("claim"):
            if node.attrs.get("kind") != "literature":
                continue
            edges = self.incoming(node.id, {"supports"})
            if all(e.attrs.get("basis") == "claimed" or self.nodes[e.source].attrs.get("evidence_quality") in {"low", "unknown"} for e in edges):
                out.append({"claim": node.id, "label": node.label, "papers": [e.source for e in edges]})
        return out

    def uncertainty_hotspots(self, allow_synthetic: bool = False, limit: int = 5) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes_of("claim"):
            if not node.attrs.get("hypothesis_claim"):
                continue
            state, _ = self.derive_state(node.id, allow_synthetic)
            weight = 1.0 if node.attrs.get("importance", "central") == "central" else 0.6
            pending = [x for x in self.experiments_for_claim(node.id) if not self.outgoing(x, {"produces"})]
            out.append(
                {
                    "claim": node.id,
                    "label": node.label,
                    "state": state,
                    "uncertainty": round(UNCERTAINTY.get(state, 1.0) * weight, 3),
                    "pending_experiments": pending,
                }
            )
        out.sort(key=lambda row: row["uncertainty"], reverse=True)
        return out[:limit]

    def summary(self, allow_synthetic: bool = False) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for node in self.nodes.values():
            if node.attrs.get("status") in INACTIVE:
                continue
            counts[node.type] = counts.get(node.type, 0) + 1
        states: Dict[str, int] = {}
        for state in self.claim_states(allow_synthetic).values():
            states[state] = states.get(state, 0) + 1
        return {"nodes": counts, "edges": len(self.edges), "claim_states": states}
