"""The hosted generator: OpenAI's Responses API, one stateless call per draft.

This is the project's **only external, paid dependency**, and it exists beside
the local Qwen/vLLM backend rather than instead of it. The Qwen profile, its
completed 48-group run and its snapshot are untouched historical evidence; what
this adds is a second generator to try, under the same prompts, the same
validators, the same thresholds, the same repair budget and the same curator
gate. Nothing about the corpus rules moves to accommodate it.

**The credential.** It is read from ``OPENAI_API_KEY`` at the moment of the
call and nowhere else. It is never written to a request record, a log line, a
raw file, an error message, a printed line, a command example or a hash. The
recorded payload is the complete request body and the Authorization header is
not part of it: the header is built inside :meth:`OpenAIResponsesBackend.send`
and never leaves it. Every error string this module produces is passed through
:func:`redact_secrets` as well, so a provider message that echoed something
secret could not carry it into a log.

**One draft per call, and no hidden retry.** ``store: false``,
``background: false``, no tools, no files, no conversation and no
``previous_response_id``: each request is stateless and self-contained. There is
no automatic retry at any level — an invisible retry is a second paid call for
the same draft. A transport failure is audited by the controller, consumes no
budget position, and is left for a person to decide about.

``store: false`` is about *state*, not retention: it keeps the response out of
retrievable Responses API application state — nothing can be fetched back by
response id or chained through ``previous_response_id`` — but it is not a
retention guarantee. Ordinary provider abuse-monitoring retention may still
apply under the account's data-control policy, and Zero Data Retention is a
property of the account rather than of this flag; nothing here claims it.

**What is charged is recorded.** A call that reached the provider and came back
is recorded even when what came back is unusable: a truncated or malformed
response is a *rejected* attempt that consumed its budget position, never a
transport failure to be silently retried. Only a failure that produced no
response at all — no network, a timeout, an authentication or rate-limit
refusal, an unavailable model — raises.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..config import ExperimentConfig
from .backends import (
    BackendError,
    BackendResponse,
    BackendUnavailable,
    LiveCallRefused,
)
from .requests import DraftRequest

__all__ = [
    "API_KEY_ENV",
    "OPENAI_AUTHORIZATION_ENV",
    "OPENAI_BACKEND",
    "OpenAIAuthError",
    "OpenAIModelUnavailable",
    "OpenAIRateLimited",
    "OpenAIResponsesBackend",
    "openai_authorization_problems",
    "openai_preflight_problems",
    "openai_responses_payload",
    "redact_secrets",
]

#: The configured backend name, and the value ``models.generator.backend`` takes.
OPENAI_BACKEND = "openai_responses"

#: The external-provider authorisation. Separate from the local-generation key
#: on purpose: permitting a run on our own GPU and permitting a paid call to a
#: third party are different decisions, and neither belongs in a config file.
OPENAI_AUTHORIZATION_ENV = "REASONSTYLE_ALLOW_OPENAI_GENERATION"

#: The only place the credential comes from. Never read outside ``send``.
API_KEY_ENV = "OPENAI_API_KEY"

#: Anything shaped like a key, scrubbed from every message this module emits.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"(?i)(\"?(?:api[_-]?key|authorization)\"?\s*[:=]\s*\"?)[A-Za-z0-9_\-\.]{8,}"),
)


class OpenAIAuthError(BackendError):
    """The provider rejected the credential. The value is never echoed."""


class OpenAIRateLimited(BackendError):
    """The provider refused for rate or quota reasons. Nothing is retried here."""


class OpenAIModelUnavailable(BackendError):
    """The pinned model is not available to this account."""


def redact_secrets(text: str, *, secret: str | None = None) -> str:
    """Remove anything credential-shaped from a message before it is shown.

    ``secret`` is the live key when the caller has it in hand; it is compared
    and removed, never stored or returned. The patterns catch the rest, so a
    provider message that echoed a token could not carry it into a log either.
    """
    if not text:
        return text
    if secret:
        text = text.replace(secret, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) + "[redacted]") if m.groups() else "[redacted]",
                           text)
    return text


def openai_authorization_problems(allow_live: bool, *,
                                  env: Mapping[str, str] | None = None) -> list[str]:
    """Why a live OpenAI call may not proceed. Empty means both yeses are present.

    Two independent decisions, neither of them in the configuration: the caller
    passed ``--send``, and ``REASONSTYLE_ALLOW_OPENAI_GENERATION=1`` is set. The
    local-generation key does **not** authorise a paid external call, and this
    key does not authorise a local one.
    """
    import os
    env = os.environ if env is None else env
    problems = []
    if not allow_live:
        problems.append("the caller did not pass --send (allow_live=True)")
    if env.get(OPENAI_AUTHORIZATION_ENV) != "1":
        problems.append(f"{OPENAI_AUTHORIZATION_ENV}=1 is not set in the environment: a paid "
                        f"call to an external provider is a separate authorisation from a "
                        f"local model run")
    return problems


def openai_preflight_problems(cfg: ExperimentConfig, *,
                              env: Mapping[str, str] | None = None) -> list[str]:
    """What must hold before the first paid call. Reads no secret value.

    The credential is checked for **presence only** — whether the variable is
    set and non-empty. Its value is never read into a variable here, never
    measured, never hashed and never printed.
    """
    import os
    env = os.environ if env is None else env
    gen = cfg.raw["models"]["generator"]
    problems = []
    if gen["backend"] != OPENAI_BACKEND:
        problems.append(f"the configuration names backend {gen['backend']!r}, "
                        f"not {OPENAI_BACKEND!r}")
        return problems
    name = gen["openai"]["api_key_env_name"]
    if not (env.get(name) or "").strip():
        problems.append(f"{name} is not set in the environment. Export it in the shell for "
                        f"the duration of the run; nothing here reads, prints or stores "
                        f"its value.")
    if not gen["model"].get("id"):
        problems.append("no generator model id is configured")
    endpoint = gen["openai"]["endpoint"]
    if not endpoint.startswith("https://"):
        problems.append(f"the endpoint must be HTTPS; got {endpoint!r}")
    return problems


def openai_responses_payload(request: DraftRequest, cfg: ExperimentConfig) -> dict[str, Any]:
    """The exact request body, built from the configuration alone.

    Everything here is non-secret and is recorded verbatim. What is *absent* is
    as deliberate as what is present: no ``top_p`` (left at the provider
    default rather than tuned alongside temperature), no ``tools``, no
    ``previous_response_id``, no ``conversation``, no attachment and no
    ``seed`` — this API offers none for this model, and a seed that was never
    sent must not be recorded as though it had been.

    Structured output is constrained provider-side through ``text.format`` with
    the template's own closed schema and ``strict: true``, so the model cannot
    answer with prose that merely resembles an object.
    """
    gen = cfg.raw["models"]["generator"]
    dec, api = gen["decoding"], gen["openai"]
    model = gen["model"]["pinned_snapshot"] or gen["model"]["id"]
    return {
        "model": model,
        "input": [{"role": "user",
                   "content": [{"type": "input_text", "text": request.prompt}]}],
        "temperature": dec["temperature"],
        "max_output_tokens": dec["max_output_tokens"],
        "reasoning": dict(dec["reasoning"]),
        "text": {"format": {"type": "json_schema", "name": "draft",
                            "schema": request.response_schema, "strict": api["strict"]}},
        "store": api["store"],
        "background": api["background"],
    }


def _usable_text(raw: dict[str, Any]) -> str | None:
    """The one structured answer, out of a Responses API reply.

    ``output_text`` is the convenience field; the output array is read when it
    is absent. Reasoning items carry no content and are skipped. Nothing is
    concatenated across several messages: a call that produced more than one
    message is not the single draft that was asked for.
    """
    text = raw.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    messages = [item for item in (raw.get("output") or [])
                if isinstance(item, dict) and item.get("type") == "message"]
    if len(messages) != 1:
        return None
    parts = [part.get("text") for part in (messages[0].get("content") or [])
             if isinstance(part, dict) and part.get("type") == "output_text"]
    parts = [p for p in parts if isinstance(p, str) and p.strip()]
    return parts[0] if len(parts) == 1 else None


def _stop_reason(raw: dict[str, Any]) -> str:
    """``stop`` only for a completed response; anything else says what happened.

    The controller treats a stop reason other than ``stop`` as a rejected
    attempt: it consumed its budget position and is recorded, never silently
    re-sent.
    """
    status = raw.get("status")
    if status == "completed":
        return "stop"
    reason = ((raw.get("incomplete_details") or {}).get("reason")
              if isinstance(raw.get("incomplete_details"), dict) else None)
    return f"{status}:{reason}" if reason else str(status)


@dataclass
class OpenAIResponsesBackend:
    """One stateless Responses API call per draft. No retry, no stored state."""

    endpoint: str | None = None
    timeout: float | None = None
    #: Injected in tests so that no test can reach a network, ever.
    _urlopen: Callable[..., Any] = field(default=urllib.request.urlopen, repr=False)
    _env: Mapping[str, str] | None = field(default=None, repr=False)

    def _environ(self) -> Mapping[str, str]:
        import os
        return os.environ if self._env is None else self._env

    def send(self, request: DraftRequest, cfg: ExperimentConfig, *,
             allow_live: bool = False) -> BackendResponse:
        env = self._environ()
        problems = openai_authorization_problems(allow_live, env=env)
        if problems:
            raise LiveCallRefused("; ".join(problems) + " — nothing was sent")

        gen = cfg.raw["models"]["generator"]
        api = gen["openai"]
        # The one moment the credential exists in this process. It is used to
        # build one header and is never returned, stored, logged or hashed.
        secret = (env.get(api["api_key_env_name"]) or "").strip()
        if not secret:
            raise BackendUnavailable(
                f"{api['api_key_env_name']} is not set in the environment — nothing was "
                f"sent. Export it in the shell for the duration of the run; its value is "
                f"never read, printed or stored by this code.")

        url = self.endpoint or api["endpoint"]
        timeout = self.timeout or api["timeout_seconds"]
        payload = openai_responses_payload(request, cfg)
        http = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"content-type": "application/json",
                     # Never recorded: the stored request payload is the body
                     # alone, and this header is built here and discarded here.
                     "authorization": f"Bearer {secret}"})
        try:
            with self._urlopen(http, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                request_id = _header(response, "x-request-id")
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc, secret=secret, model=payload["model"]) from None
        except urllib.error.URLError as exc:
            raise BackendUnavailable(redact_secrets(
                f"could not reach {url}: {exc.reason}. Nothing was sent again: there is no "
                f"automatic retry, because a retry is a second paid call.",
                secret=secret)) from None
        except TimeoutError as exc:
            raise BackendUnavailable(redact_secrets(
                f"the request to {url} timed out after {timeout}s: {exc}. It is not retried "
                f"automatically; the response may or may not have been billed, so a repeat "
                f"is your decision.", secret=secret)) from None
        return self._response_from(body, request_id=request_id, url=url, secret=secret,
                                   requested_model=payload["model"])

    # -- failures that produced no response ---------------------------------

    def _http_error(self, exc: urllib.error.HTTPError, *, secret: str,
                    model: str) -> BackendError:
        """Map a provider error to a typed failure that cannot carry the key."""
        try:
            detail = exc.read().decode("utf-8", "replace")[:500]
        except Exception:                                   # pragma: no cover - defensive
            detail = ""
        detail = redact_secrets(detail, secret=secret)
        if exc.code in (401, 403):
            return OpenAIAuthError(
                f"HTTP {exc.code}: the provider rejected the credential. Check that "
                f"{API_KEY_ENV} is exported and valid for this account; its value is never "
                f"read, printed or stored here. Provider said: {detail}")
        if exc.code == 429:
            return OpenAIRateLimited(
                f"HTTP 429: rate or quota limit. Nothing was retried — an automatic retry "
                f"would be a second paid call for the same draft. Provider said: {detail}")
        if exc.code == 404 or "model_not_found" in detail:
            return OpenAIModelUnavailable(
                f"HTTP {exc.code}: the account cannot use model {model!r}. Do not substitute "
                f"another model; report it. Provider said: {detail}")
        return BackendError(f"HTTP {exc.code} from the Responses API: {detail}")

    # -- a response that arrived ---------------------------------------------

    def _response_from(self, body: str, *, request_id: str | None, url: str, secret: str,
                       requested_model: str) -> BackendResponse:
        """Turn a 200 into a recorded attempt, usable or not.

        A response that arrived was paid for, so it is always returned as an
        attempt. If it is truncated or malformed, the stop reason and the
        unusable content say so and the controller records a *rejected*
        attempt, which consumes its budget position and is never re-sent.
        """
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as exc:
            raise BackendError(redact_secrets(
                f"the provider returned a body that is not JSON: {exc}",
                secret=secret)) from None
        if not isinstance(raw, dict):
            raise BackendError("the provider returned a JSON value that is not an object")

        returned = raw.get("model")
        if returned and returned != requested_model:
            if not str(returned).startswith(requested_model):
                # A different model is not a more specific pin; nothing silently
                # follows it, and the call is not repeated either.
                raise OpenAIModelUnavailable(
                    f"asked for {requested_model!r} but the provider served {returned!r}. "
                    f"Nothing was retried. Report this before any further call.")
            # A more specifically pinned snapshot of the same model: recorded and
            # surfaced for approval, never written into the configuration here.

        stop_reason = _stop_reason(raw)
        text = _usable_text(raw)
        provider = {
            "provider": "openai",
            "api": "responses",
            "endpoint": url,
            "response_id": raw.get("id"),
            "request_id": request_id,
            "requested_model": requested_model,
            "returned_model": returned,
            "snapshot_more_specific_than_requested": bool(
                returned and returned != requested_model
                and str(returned).startswith(requested_model)),
            "status": raw.get("status"),
            "incomplete_details": raw.get("incomplete_details"),
            "store": raw.get("store", False),
            "background": raw.get("background", False),
            "created_at": raw.get("created_at"),
            "service_tier": raw.get("service_tier"),
        }
        content: Any = None
        if text is not None:
            try:
                content = json.loads(text)
            except json.JSONDecodeError:
                # Kept, not patched: the controller rejects it under the
                # template's own schema and records the attempt as spent.
                content = text
        return BackendResponse(content=content, model_returned=returned,
                               stop_reason=stop_reason, usage=raw.get("usage"), raw=raw,
                               provider_meta=provider)


def _header(response: Any, name: str) -> str | None:
    """One response header, when the transport exposes them."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get(name)
    except Exception:                                       # pragma: no cover - defensive
        return None
