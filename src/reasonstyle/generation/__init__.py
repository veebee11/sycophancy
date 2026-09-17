"""Drafting the corpus: allocation, hashed prompts, requests, logging.

Nothing in this package calls a model by itself. A backend is passed in, and
the only backend that can reach a network refuses unless the configuration
enables live calls *and* the caller asks for one explicitly.

The flow is deliberately file-based, so that every request can be read before
it is sent and every response is kept exactly as it arrived:

    allocation  ->  requests (files)  ->  [backend]  ->  responses (files)
                                                     ->  log + corpus records

:mod:`provenance` closes the loop: it rebuilds each request from the curated
brief, the allocation and the hashed template, and checks that the request
which was actually sent matches. That is what makes independence from the
source datasets a procedural guarantee rather than a claim.
"""

from .allocation import (
    AllocationError,
    GroupAllocation,
    MarkerAllocation,
    allocate_markers,
    allocation_problems,
    load_allocation,
    save_allocation,
)
from .approvals import (
    APPROVED,
    ApprovalError,
    ScenarioApproval,
    approval_status,
    gate_problems,
    load_approvals,
    save_approvals,
)
from .assemble import (
    AssemblyError,
    AssemblyRefused,
    ManualCorrection,
    assemble_scenario,
    correction_problems,
    load_corrections,
    save_corrections,
)
from .backends import (
    AUTHORIZATION_ENV,
    BackendError,
    BackendUnavailable,
    FakeBackend,
    LiveCallRefused,
    VLLMOpenAIBackend,
    authorization_problems,
    offline_problems,
    vllm_payload,
)
from .environment import (
    CachedModel,
    ModelNotCached,
    RunEnvironment,
    describe_run,
    load_server_runtime,
    missing_weights,
    resolve_cached_model,
    revision_agreement,
    server_settings_problems,
)
from .log import GenerationLog, LogEntry
from .requests import (
    DraftRequest,
    RequestError,
    ResponseRejected,
    group_request,
    parse_response,
    repair_request,
    scenario_request,
    write_request,
)
from .pipeline import (
    ABORTED,
    ACCEPTED,
    Attempt,
    CallStore,
    NEEDS_MANUAL_REVIEW,
    PipelineAbort,
    StageResult,
    draft_group,
    draft_scenario,
    run_pilot,
)
from .redraft import (
    RedraftTarget,
    current_scenarios,
    redraft_request,
    redraft_targets,
    redraft_template_sha256,
    run_redraft_stage,
)
from .provenance import ProvenanceMismatch, check_request_provenance

__all__ = [
    "ABORTED",
    "APPROVED",
    "ApprovalError",
    "AssemblyError",
    "AssemblyRefused",
    "ManualCorrection",
    "ScenarioApproval",
    "approval_status",
    "assemble_scenario",
    "correction_problems",
    "gate_problems",
    "load_approvals",
    "load_corrections",
    "save_approvals",
    "save_corrections",
    "ACCEPTED",
    "AUTHORIZATION_ENV",
    "AllocationError",
    "Attempt",
    "CallStore",
    "NEEDS_MANUAL_REVIEW",
    "PipelineAbort",
    "RedraftTarget",
    "current_scenarios",
    "redraft_request",
    "redraft_targets",
    "redraft_template_sha256",
    "run_redraft_stage",
    "StageResult",
    "draft_group",
    "draft_scenario",
    "run_pilot",
    "BackendError",
    "BackendUnavailable",
    "CachedModel",
    "DraftRequest",
    "FakeBackend",
    "GenerationLog",
    "GroupAllocation",
    "LiveCallRefused",
    "LogEntry",
    "MarkerAllocation",
    "ModelNotCached",
    "ProvenanceMismatch",
    "RequestError",
    "ResponseRejected",
    "RunEnvironment",
    "VLLMOpenAIBackend",
    "allocate_markers",
    "allocation_problems",
    "authorization_problems",
    "check_request_provenance",
    "describe_run",
    "group_request",
    "load_allocation",
    "load_server_runtime",
    "missing_weights",
    "offline_problems",
    "parse_response",
    "repair_request",
    "resolve_cached_model",
    "revision_agreement",
    "server_settings_problems",
    "save_allocation",
    "scenario_request",
    "vllm_payload",
    "write_request",
]
