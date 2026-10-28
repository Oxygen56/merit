"""Merit: controlled evaluation for agent improvement methods."""

from .adapters import ProposalContext, PythonOptimizerAdapter
from .agent_contract import (
    AgentContractManifest,
    AgentExecutor,
    AgentOptimizerAdapter,
    AgentProposal,
    AgentProposalContext,
    AgentResponse,
    AgentSpec,
    AgentTask,
    Usage,
    evaluate_agent,
    run_agent_contract,
    tasks_digest,
)
from .agent_methods import (
    AgentNoChangeOptimizer,
    FailureFeedbackProposer,
    ModelReflectionProposer,
    RandomPromptEditor,
    accept_always,
)
from .agent_pair import AgentPairedExperiment, AgentPairedManifest, AgentTaskBundle, run_agent_paired
from .budget import Budget, BudgetExceeded
from .evidence import (
    EVIDENCE_SCHEMA_VERSION,
    compress_evidence,
    evidence_schema,
    read_evidence,
    validate_evidence_envelope,
    validate_result_directory,
)
from .gepa_optimizer import GepaOptimizer
from .meta_plasticity import (
    ImproverSnapshot,
    ImproverTrajectory,
    MetaPlasticityExperiment,
    MetaPlasticityManifest,
    PlasticityProposer,
    candidate_evaluations_to_threshold,
    make_meta_plasticity_manifest,
    run_meta_plasticity,
)
from .model import AgentArtifact, Experiment, ExperimentManifest, Task
from .openai_executor import OpenAICompatibleExecutor, TokenPricing
from .ollama_executor import OllamaExecutor
from .langgraph_executor import LangGraphToolExecutor
from .optimizers import MemoryResetOptimizer, NoChangeOptimizer, RandomEditOptimizer
from .runner import PairRunner
from .transfer import TaskFamily, TransferExperiment, TransferManifest, run_transfer_evaluation

__all__ = [
    "AgentArtifact",
    "AgentContractManifest",
    "AgentExecutor",
    "AgentNoChangeOptimizer",
    "AgentOptimizerAdapter",
    "AgentPairedExperiment",
    "AgentPairedManifest",
    "AgentProposal",
    "AgentProposalContext",
    "AgentResponse",
    "AgentSpec",
    "AgentTask",
    "AgentTaskBundle",
    "Budget",
    "BudgetExceeded",
    "EVIDENCE_SCHEMA_VERSION",
    "Experiment",
    "ExperimentManifest",
    "FailureFeedbackProposer",
    "GepaOptimizer",
    "ImproverSnapshot",
    "ImproverTrajectory",
    "LangGraphToolExecutor",
    "MemoryResetOptimizer",
    "MetaPlasticityExperiment",
    "MetaPlasticityManifest",
    "ModelReflectionProposer",
    "NoChangeOptimizer",
    "OllamaExecutor",
    "OpenAICompatibleExecutor",
    "PairRunner",
    "PlasticityProposer",
    "ProposalContext",
    "PythonOptimizerAdapter",
    "RandomEditOptimizer",
    "RandomPromptEditor",
    "Task",
    "TaskFamily",
    "TokenPricing",
    "TransferExperiment",
    "TransferManifest",
    "Usage",
    "accept_always",
    "candidate_evaluations_to_threshold",
    "compress_evidence",
    "evaluate_agent",
    "evidence_schema",
    "make_meta_plasticity_manifest",
    "read_evidence",
    "run_agent_contract",
    "run_agent_paired",
    "run_meta_plasticity",
    "run_transfer_evaluation",
    "tasks_digest",
    "validate_evidence_envelope",
    "validate_result_directory",
]

__version__ = "0.1.0"
