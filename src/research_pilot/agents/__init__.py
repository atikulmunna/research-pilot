from .base import Agent, AgentDeps
from .experiment_designer import ExperimentDesigner
from .experiment_engineer import ExperimentEngineer
from .field_scout import FieldScout
from .gap_novelty import GapNoveltyAnalyst
from .hypothesis import HypothesisDesigner
from .interpreter import ScientificInterpreter
from .literature import LiteratureIntelligence
from .paper_architect import PaperArchitect
from .planner import ResearchPlanner
from .proposal_critic import ProposalCritic
from .quant_analyst import QuantitativeAnalyst
from .red_team import RedTeamReviewer

AGENT_CLASSES = (
    FieldScout,
    LiteratureIntelligence,
    GapNoveltyAnalyst,
    ProposalCritic,
    HypothesisDesigner,
    ExperimentDesigner,
    ExperimentEngineer,
    QuantitativeAnalyst,
    ScientificInterpreter,
    ResearchPlanner,
    PaperArchitect,
    RedTeamReviewer,
)

__all__ = ["AGENT_CLASSES", "Agent", "AgentDeps"]
