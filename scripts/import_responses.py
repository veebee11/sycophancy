"""Import responses, validate their shape, log every call, store the raw traffic.

    uv run python scripts/import_responses.py --config configs/experiment.yaml \
        --topics data/topics/pilot_topics.yaml --requests data/pilot/requests \
        --responses data/pilot/responses

A response file is named after the request it answers (``<call_id>.json``) and
holds what the provider returned. Each response is checked against the
template's schema before it is stored: a response that does not fit is rejected
and logged as rejected, never patched into shape.

Scenario texts land in ``data/pilot/scenarios.yaml``; group drafts land in
``data/pilot/group_drafts.yaml``. Assembling them into corpus records, and
validating those records, is a separate step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from reasonstyle.config import load_config
from reasonstyle.corpus.topics import load_topic_bank
from reasonstyle.generation import GenerationLog, ResponseRejected, parse_response
from reasonstyle.generation.log import LogEntry, utc_now
from reasonstyle.hashing import content_hash
from reasonstyle.generation.allocation import AllocationError, load_allocation


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_yaml(path: Path, data: dict, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(data, sort_keys=True, allow_unicode=True, width=100),
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--topics", required=True)
    ap.add_argument("--requests", default="data/pilot/requests")
    ap.add_argument("--responses", default="data/pilot/responses")
    ap.add_argument("--allocation", default="data/pilot/marker_allocation.yaml")
    ap.add_argument("--scenarios-out", default="data/pilot/scenarios.yaml")
    ap.add_argument("--drafts-out", default="data/pilot/group_drafts.yaml")
    ap.add_argument("--log", default="data/pilot/generation_log.jsonl")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    bank = load_topic_bank(args.topics)
    bank_hash = content_hash(bank.model_dump(mode="json"))
    try:
        alloc_hash = load_allocation(args.allocation).content_hash
    except (AllocationError, FileNotFoundError):
        alloc_hash = None

    log = GenerationLog(args.log)
    scenarios = _load_yaml(Path(args.scenarios_out))
    drafts = _load_yaml(Path(args.drafts_out))

    imported = rejected = missing = 0
    for request_path in sorted(Path(args.requests).glob("*.json")):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response_path = Path(args.responses) / f"{request['call_id']}.json"
        if not response_path.exists():
            missing += 1
            continue
        stored = json.loads(response_path.read_text(encoding="utf-8"))
        content = stored.get("content", stored)

        status, error, fields = "ok", None, None
        try:
            fields = parse_response(_request_shim(request), content)
        except ResponseRejected as exc:
            status, error, rejected = "rejected", str(exc), rejected + 1
        else:
            imported += 1
            key = f"{request['decision_id']}_v{request['variant_id']}"
            if request["kind"] == "scenario":
                scenarios[key] = fields["scenario_text"]
            else:
                drafts.setdefault(key, {})[request["supported_option"]] = {
                    "bodies": fields,
                    "attempt": request["attempt"],
                    **request["context"],
                }

        log.store_raw(request["call_id"], stored.get("request_payload", {}), stored,
                      prompt=request["prompt"])
        log.append(LogEntry(
            call_id=request["call_id"], kind=request["kind"], attempt=request["attempt"],
            decision_id=request["decision_id"], variant_id=request["variant_id"],
            supported_option=request["supported_option"],
            template_name=request["template_name"], template_sha256=request["template_sha256"],
            prompt_sha256=request["prompt_sha256"],
            model=cfg.raw["models"]["generator"]["model"],
            model_returned=stored.get("model"),
            request_fields=stored.get("request_fields", {}),
            config_content_hash=cfg.content_hash, topic_bank_content_hash=bank_hash,
            allocation_content_hash=alloc_hash,
            response_sha256=GenerationLog.response_digest(content),
            stop_reason=stored.get("stop_reason"), usage=stored.get("usage"),
            status=status, error=error, generated_at=utc_now()))

    _save_yaml(Path(args.scenarios_out), scenarios,
               "# Drafted scenario texts. GENERATED from the imported responses.\n")
    _save_yaml(Path(args.drafts_out), drafts,
               "# Drafted counterargument groups. GENERATED from the imported responses.\n")

    print(f"imported {imported}, rejected {rejected}, awaiting response {missing}")
    print(f"  log {args.log}, raw traffic {log.raw_dir}/")
    return 1 if rejected else 0


class _request_shim:
    """Just enough of a DraftRequest for :func:`parse_response`: the stored
    request file is the authority, and it is never rebuilt from the config
    here — that is what the provenance check is for."""

    def __init__(self, data: dict):
        self.response_schema = data["response_schema"]


if __name__ == "__main__":
    raise SystemExit(main())
