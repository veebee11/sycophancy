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
from .provenance import ProvenanceMismatch, check_request_provenance

__all__ = [
    "AUTHORIZATION_ENV",
    "AllocationError",
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
