"""Backends: a fake one for tests, and a local vLLM endpoint.

The generator is an open-weights model hosted on a lab GPU server and reached
over the loopback interface through vLLM's OpenAI-compatible API. There is no
external service, no paid call and no credential: nothing here reads, stores or
sends an API key.

**Running needs two independent yeses, neither of them in the experiment
config**: the caller passes ``allow_live=True`` (from an explicit ``--send``),
and ``REASONSTYLE_ALLOW_LOCAL_GENERATION=1`` is set in the environment. Reading
or sharing the configuration therefore cannot by itself enable a model run. The
payload can always be built and inspected without either.

Extending this later — an in-process Transformers backend, say — means adding a
class with the same ``send`` signature. Requests, the response schema, parsing,
hashing, the repair budget and the log are backend-independent and do not move.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import ExperimentConfig
from .requests import DraftRequest

__all__ = [
    "AUTHORIZATION_ENV",
    "BackendError",
    "BackendResponse",
    "BackendUnavailable",
    "FakeBackend",
    "LiveCallRefused",
    "VLLMOpenAIBackend",
    "authorization_problems",
    "vllm_payload",
]

#: Set to "1" in the shell for the duration of an approved run. Deliberately not
#: a config key: a scientific configuration should not carry permission to run.
AUTHORIZATION_ENV = "REASONSTYLE_ALLOW_LOCAL_GENERATION"


class BackendError(RuntimeError):
    """The backend could not produce a response."""


class LiveCallRefused(BackendError):
    """A run was attempted without both authorisations."""


class BackendUnavailable(BackendError):
    """The local server is not reachable, or the environment is not set up."""


@dataclass(frozen=True, slots=True)
class BackendResponse:
    content: dict[str, Any]          # the structured object the model returned
    model_returned: str | None = None
    stop_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None                  # the server's whole response, verbatim


def authorization_problems(allow_live: bool, *, env: dict[str, str] | None = None) -> list[str]:
    """Why a run may not proceed. Empty means both yeses are present."""
    env = os.environ if env is None else env
    problems = []
    if not allow_live:
        problems.append("the caller did not pass --send (allow_live=True)")
    if env.get(AUTHORIZATION_ENV) != "1":
        problems.append(f"{AUTHORIZATION_ENV}=1 is not set in the environment")
    return problems


def offline_problems(cfg: ExperimentConfig, *, env: dict[str, str] | None = None) -> list[str]:
    """Offline mode must be on, or a missing model becomes a silent download."""
    env = os.environ if env is None else env
    required = cfg.raw["models"]["generator"]["vllm"]["require_offline_env"]
    return [f"{key}={value} is required so that a missing model is an error, not a download"
            for key, value in required.items() if env.get(key) != value]


def vllm_payload(request: DraftRequest, cfg: ExperimentConfig) -> dict[str, Any]:
    """The exact request body, built from the configuration alone.

    Structured output is constrained server-side: the template's response schema
    is passed as a JSON-schema response format, so the model cannot return prose
    that merely resembles an object. The prompt is one user message; the model's
    own chat template is applied by the server.
    """
    gen = cfg.raw["models"]["generator"]
    dec = gen["decoding"]
    payload: dict[str, Any] = {
        "model": gen["model"]["repo_id"],
        "messages": [{"role": "user", "content": request.prompt}],
        "temperature": dec["temperature"],
        "top_p": dec["top_p"],
        "max_tokens": dec["max_tokens"],
        "n": dec["n"],
        "seed": dec["seed"],
        # Sent explicitly at neutral values: nothing is left to a server default.
        "top_k": dec["top_k"],
        "min_p": dec["min_p"],
        "repetition_penalty": dec["repetition_penalty"],
        "presence_penalty": dec["presence_penalty"],
        "frequency_penalty": dec["frequency_penalty"],
    }
    if gen["vllm"]["guided_json"]:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "draft", "schema": request.response_schema,
                            "strict": True},
        }
    if dec["thinking"] == "disabled":
        # Qwen3 switches thinking off through its chat template; vLLM passes
        # template arguments through chat_template_kwargs.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


class FakeBackend:
    """Deterministic stand-in. Tests and dry runs never touch a server.

    ``responder`` maps a request to the object a model would have returned.
    Every call is recorded, so a test can assert exactly what was asked — and
    that nothing was asked twice.
    """

    def __init__(self, responder: Callable[[DraftRequest], Any]):
        self._responder = responder
        self.calls: list[DraftRequest] = []

    def send(self, request: DraftRequest, cfg: ExperimentConfig, *,
             allow_live: bool = False) -> BackendResponse:
        self.calls.append(request)
        content = self._responder(request)
        if isinstance(content, BaseException):
            raise BackendError(str(content))
        return BackendResponse(content=content, model_returned="fake-backend",
                               stop_reason="stop",
                               usage={"prompt_tokens": len(request.prompt.split()),
                                      "completion_tokens": sum(
                                          len(str(v).split()) for v in content.values())
                                      if isinstance(content, dict) else 0},
                               raw={"fake": True, "content": content})


@dataclass
class VLLMOpenAIBackend:
    """A vLLM server on localhost, spoken to over its OpenAI-compatible API."""

    base_url: str | None = None
    timeout: float | None = None
    _urlopen: Callable[..., Any] = field(default=urllib.request.urlopen, repr=False)

    def _endpoint(self, cfg: ExperimentConfig, suffix: str) -> str:
        base = (self.base_url or cfg.raw["models"]["generator"]["vllm"]["base_url"]).rstrip("/")
        return f"{base}{suffix}"

    def _get(self, url: str, timeout: float) -> Any:
        try:
            with self._urlopen(urllib.request.Request(url, method="GET"),
                               timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:                   # pragma: no cover - network
            raise BackendUnavailable(f"no vLLM server at {url}: {exc.reason}") from exc

    def served_models(self, cfg: ExperimentConfig) -> list[dict[str, Any]]:
        """What the server says it is serving. Used to check the loaded model
        against the one the configuration names."""
        timeout = self.timeout or cfg.raw["models"]["generator"]["vllm"]["timeout_seconds"]
        return self._get(self._endpoint(cfg, "/models"), timeout).get("data", [])

    def send(self, request: DraftRequest, cfg: ExperimentConfig, *,
             allow_live: bool = False) -> BackendResponse:
        problems = authorization_problems(allow_live) + offline_problems(cfg)
        if problems:
            raise LiveCallRefused("; ".join(problems) + " — nothing was sent")

        gen = cfg.raw["models"]["generator"]
        timeout = self.timeout or gen["vllm"]["timeout_seconds"]
        payload = vllm_payload(request, cfg)
        url = self._endpoint(cfg, "/chat/completions")
        body = json.dumps(payload).encode("utf-8")
        http = urllib.request.Request(url, data=body, method="POST",
                                      headers={"content-type": "application/json"})
        try:
            with self._urlopen(http, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:                  # pragma: no cover - network
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise BackendError(f"HTTP {exc.code} from the local server: {detail}") from exc
        except urllib.error.URLError as exc:                   # pragma: no cover - network
            raise BackendUnavailable(
                f"no vLLM server at {url}: {exc.reason}. Start it on the GPU host first."
            ) from exc
        return self._response_from(raw)

    @staticmethod
    def _response_from(raw: dict[str, Any]) -> BackendResponse:
        choices = raw.get("choices") or []
        if not choices:
            raise BackendError(f"the server returned no choices: {str(raw)[:300]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not text:
            raise BackendError(
                f"the response carried no content (finish_reason="
                f"{choices[0].get('finish_reason')!r})")
        try:
            content = json.loads(text)
        except json.JSONDecodeError as exc:
            # Kept as an error: a response that is not the requested object is
            # rejected and logged, never patched into shape or retried here.
            raise BackendError(f"the response was not the requested JSON object: {exc}") from exc
        return BackendResponse(content=content, model_returned=raw.get("model"),
                               stop_reason=choices[0].get("finish_reason"),
                               usage=raw.get("usage"), raw=raw)
