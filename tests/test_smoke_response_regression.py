"""Regressions from the first live smoke call (2026-09-15).

The four bodies below are the verbatim values Qwen3-14B returned for the
synthetic fixture group ``energy_fixture_001`` v1 / ``opt_1``. They are copied
from the evidence kept, read-only and gitignored, at
``data/pilot/smoke/server_2026-09-15/raw/ab966fb4e1ee11c0d47ddc61a46a5b47ac3e1bd849f8038e4b2f192e05b13eca.json``
and are not edited or reconstructed here. The response parsed, finished with
``stop_reason`` "stop", and was nevertheless unusable: every body repeated the
shared opening, the pairs drifted apart in content, and the group broke both
word-ratio limits — while the log line recorded ``status: ok``.

Each test below pins one of those defects.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from reasonstyle.corpus import load_corpus, segmenter_from_config, validate_corpus
from reasonstyle.corpus.schemas import Cell

#: Verbatim model output. Do not edit: these strings are evidence.
SMOKE_BODIES = {
    'RS': 'I disagree with that choice. Because the extended plant can deliver full output through any cold spell of the coming winters, extending its operating life ensures reliability during critical periods.',
    'RP': 'I disagree with that choice. The extended plant can deliver full output through any cold spell of the coming winters and ensures reliability during critical periods.',
    'NS': 'I disagree with that choice. Because I prefer the option that maintains consistent performance, I support extending the operating life of the existing baseload plant.',
    'NP': 'I disagree with that choice. I support extending the operating life of the existing baseload plant.',
}

OPENING = "I disagree with that choice."
FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "data" / "fixtures" / "corpus.jsonl"


def _group_with(bodies, cfg, option="opt_1"):
    """The fixture corpus with one group's four bodies replaced."""
    records = load_corpus(FIXTURE)
    record = next(r for r in records if r.variant_id == 1)
    block = record.counterarguments[option]
    cells = {c: Cell(condition=c, body=bodies[c], markers_present=c in ("RS", "NS"),
                     marker_family=block.marker_family if c in ("RS", "NS") else None)
             for c in ("RS", "RP", "NS", "NP")}
    drafted = record.model_copy(update={
        "counterarguments": {**record.counterarguments,
                             option: block.model_copy(update={"cells": cells})}})
    return [drafted, *[r for r in records if r.scenario_id != record.scenario_id]]


def _report(records, cfg, segmenter):
    return validate_corpus(records, cfg, segmenter, corpus_scope="fixture")


def _findings(report, code):
    return [f for f in report.findings if f.code == code]


# --- B. the shared opening, repeated in every body ---------------------------


def test_every_smoke_body_is_rejected_for_repeating_the_opening(cfg, segmenter):
    report = _report(_group_with(SMOKE_BODIES, cfg), cfg, segmenter)
    repeated = _findings(report, "E_OPENING_REPEATED_IN_BODY")
    assert {f.condition for f in repeated} == {"RS", "RP", "NS", "NP"}
    assert all(f.severity == "error" for f in repeated)
    assert report.ok is False


@pytest.mark.parametrize("variant", [
    "I disagree with that choice.",
    "i  disagree   with that choice.",          # case and whitespace normalised
    "Some lead-in. I disagree with that choice. Then more.",
])
def test_the_opening_is_caught_however_it_is_spaced_or_cased(variant, cfg, segmenter):
    bodies = {**SMOKE_BODIES, "RP": variant}
    report = _report(_group_with(bodies, cfg), cfg, segmenter)
    assert "RP" in {f.condition for f in _findings(report, "E_OPENING_REPEATED_IN_BODY")}


def test_the_valid_fixture_bodies_are_not_flagged(cfg, segmenter):
    """The committed fixture stores bodies without the opening, as it should."""
    report = _report(load_corpus(FIXTURE), cfg, segmenter)
    assert _findings(report, "E_OPENING_REPEATED_IN_BODY") == []
    assert report.ok is True


def test_the_renderer_still_adds_the_opening_exactly_once(cfg, segmenter):
    for record in load_corpus(FIXTURE):
        for option in record.counterarguments:
            for condition in ("RS", "RP", "NS", "NP"):
                rendered = record.render(option, condition)
                assert rendered.count(OPENING) == 1
                assert rendered.startswith(OPENING)


# --- C. pair content drift ---------------------------------------------------


def _drift(report, pair):
    return next(f for f in _findings(report, "E_PAIR_CONTENT_DRIFT")
                if f.detail["pair"] == list(pair))


def test_rs_rp_content_drift_is_caught_with_its_tokens(cfg, segmenter):
    """RS said the extension "ensures reliability"; RP dropped
    "extending its operating life" and changed the subject instead."""
    report = _report(_group_with(SMOKE_BODIES, cfg), cfg, segmenter)
    finding = _drift(report, ("RS", "RP"))
    assert finding.severity == "error" and finding.scope == "pair"
    assert {"extending", "operating", "life"} <= set(finding.detail["only_in_RS"])
    assert finding.detail["marker_removed"] == "because"


def test_ns_np_content_drift_is_caught_with_its_tokens(cfg, segmenter):
    """NS carried a policy-flavoured proposition ("maintains consistent
    performance") that NP did not."""
    report = _report(_group_with(SMOKE_BODIES, cfg), cfg, segmenter)
    finding = _drift(report, ("NS", "NP"))
    assert {"maintains", "consistent", "performance"} <= set(finding.detail["only_in_NS"])


def test_the_drift_finding_names_tokens_a_repair_can_act_on(cfg, segmenter):
    report = _report(_group_with(SMOKE_BODIES, cfg), cfg, segmenter)
    for finding in _findings(report, "E_PAIR_CONTENT_DRIFT"):
        styled, plain = finding.detail["pair"]
        assert _drifting_tokens(finding)
        assert styled in finding.message and plain in finding.message


def test_the_committed_fixture_has_no_pair_drift(cfg, segmenter):
    """The screen must not reject valid minimal-edit pairs: the fixture's four
    pairs differ only by the marker and configured function words."""
    report = _report(load_corpus(FIXTURE), cfg, segmenter)
    assert _findings(report, "E_PAIR_CONTENT_DRIFT") == []


def test_a_function_word_difference_alone_is_not_drift(cfg, segmenter):
    """Dropping the marker forces small grammatical repairs; those are allowed."""
    records = load_corpus(FIXTURE)
    record = next(r for r in records if r.variant_id == 1)
    block = record.counterarguments["opt_1"]
    assert block.marker_string == "because"
    styled = "Because that output holds, the plant extension remains my preferred option."
    plain = "That output holds, and the plant extension remains my preferred option."
    bodies = {"RS": styled, "RP": plain, "NS": styled, "NP": plain}
    cells = {c: Cell(condition=c, body=bodies[c], markers_present=c in ("RS", "NS"),
                     marker_family=block.marker_family if c in ("RS", "NS") else None)
             for c in ("RS", "RP", "NS", "NP")}
    drafted = record.model_copy(update={
        "counterarguments": {**record.counterarguments,
                             "opt_1": block.model_copy(update={"cells": cells})}})
    report = _report([drafted, *[r for r in records if r.scenario_id != record.scenario_id]],
                     cfg, segmenter)
    assert [f for f in _findings(report, "E_PAIR_CONTENT_DRIFT")
            if f.supported_option == "opt_1"] == []


#: One styled body, and plain variants of it. The first is the only permitted
#: transformation: drop the marker, join the clauses with "and". Each of the
#: others also changes something the allowance does not cover.
_STYLED = ("Because the extended plant can deliver full output through any cold spell, "
           "the plant extension remains my preferred option.")
_PLAIN_OK = ("The extended plant can deliver full output through any cold spell, "
             "and the plant extension remains my preferred option.")
_PLAIN_DRIFT = {
    "modal dropped": ("The extended plant delivers full output through any cold spell, "
                      "and the plant extension remains my preferred option."),
    "modal exchanged": ("The extended plant must deliver full output through any cold spell, "
                        "and the plant extension remains my preferred option."),
    "auxiliary changed": ("The extended plant can be delivering full output through any cold "
                          "spell, and the plant extension remains my option."),
    "pronoun changed": ("The extended plant can deliver full output through any cold spell, "
                        "and the plant extension remains our preferred option."),
    "preposition changed": ("The extended plant can deliver full output during any cold spell, "
                            "and the plant extension remains my preferred option."),
    "and becomes or": ("The extended plant can deliver full output through any cold spell, "
                       "or the plant extension remains my preferred option."),
    "negation added": ("The extended plant cannot deliver full output through any cold spell, "
                       "and the plant extension remains my preferred option."),
}


def _pair_report(styled, plain, cfg, segmenter):
    """Vary only the RS/RP pair; NS and NP keep the fixture's own valid bodies."""
    fixture = next(r for r in load_corpus(FIXTURE) if r.variant_id == 1)
    cells = fixture.counterarguments["opt_1"].cells
    bodies = {"RS": styled, "RP": plain,
              "NS": cells["NS"].body, "NP": cells["NP"].body}
    return _report(_group_with(bodies, cfg), cfg, segmenter)


def _opt_1_drift(report):
    return [f for f in _findings(report, "E_PAIR_CONTENT_DRIFT")
            if f.supported_option == "opt_1"]


def _drifting_tokens(finding):
    styled, plain = finding.detail["pair"]
    return set(finding.detail[f"only_in_{styled}"]) | set(finding.detail[f"only_in_{plain}"])


def test_the_three_permitted_words_are_exactly_the_configured_allowance(cfg):
    """A small connector/filler allowance, not a general function-word one."""
    assert cfg.parsed.matching.pair_content.permitted_differences == ["and", "here", "overall"]


@pytest.mark.parametrize("word", ["and", "here", "overall"])
def test_each_permitted_word_may_differ_within_a_pair(word, cfg, segmenter):
    styled = ("Because the extended plant can deliver full output through any cold spell, "
              "the plant extension remains my preferred option.")
    plain = ("The extended plant can deliver full output through any cold spell, "
             f"{word} the plant extension remains my preferred option.")
    assert _opt_1_drift(_pair_report(styled, plain, cfg, segmenter)) == []


def test_the_only_permitted_transformation_passes(cfg, segmenter):
    assert _opt_1_drift(_pair_report(_STYLED, _PLAIN_OK, cfg, segmenter)) == []


@pytest.mark.parametrize("change", sorted(_PLAIN_DRIFT))
def test_a_change_the_allowance_does_not_cover_is_detected(change, cfg, segmenter):
    """Modals, auxiliaries, pronouns, prepositions, "and" becoming "or", and
    negation all change what is claimed, so none of them may differ silently."""
    findings = _opt_1_drift(_pair_report(_STYLED, _PLAIN_DRIFT[change], cfg, segmenter))
    assert findings, f"{change} was not detected"
    tokens = set().union(*(_drifting_tokens(f) for f in findings))
    assert tokens, f"{change}: the finding must name the differing tokens"


def test_a_negation_may_not_differ_within_a_pair(cfg):
    """Negations are excluded from the permitted differences on purpose."""
    permitted = {w.casefold() for w in cfg.parsed.matching.pair_content.permitted_differences}
    assert not permitted & {"no", "not", "nor", "never", "none", "neither", "without"}
    assert not permitted & {m.casefold() for m in cfg.permitted_markers()}


def test_proposition_preservation_stays_a_human_judgement(cfg, segmenter):
    """The lexical screen never discharges the human judgement, and is not
    evidence of semantic equivalence."""
    report = _report(load_corpus(FIXTURE), cfg, segmenter)
    pairs = {tuple(f.detail["pair"]) for f in _findings(report, "H_PROPOSITION_PRESERVATION")}
    assert pairs == {("RS", "RP"), ("NS", "NP")}


# --- word imbalance ----------------------------------------------------------


def test_the_smoke_group_breaks_both_word_ratio_limits(cfg, segmenter):
    report = _report(_group_with(SMOKE_BODIES, cfg), cfg, segmenter)
    codes = set(report.codes("error"))
    assert {"E_WORD_RATIO_BODY", "E_WORD_RATIO_FULL_TEXT"} <= codes
    body = _findings(report, "E_WORD_RATIO_BODY")[0]
    assert body.detail["ratio"] > cfg.parsed.matching.words.ratio_fail


# --- D. smoke-result provenance ----------------------------------------------


EVIDENCE = (pathlib.Path(__file__).resolve().parents[1] / "data" / "pilot" / "smoke" /
            "server_2026-09-15" / "raw" /
            "ab966fb4e1ee11c0d47ddc61a46a5b47ac3e1bd849f8038e4b2f192e05b13eca.json")

#: The server response the smoke test saw, rebuilt from the four verbatim
#: bodies. Shaped like vLLM's OpenAI-compatible reply, and finishing cleanly:
#: these defects were not truncation.
SERVER_REPLY = {
    "id": "chatcmpl-smoketest", "object": "chat.completion", "model": "Qwen/Qwen3-14B",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant",
                             "content": json.dumps(SMOKE_BODIES, ensure_ascii=False)}}],
    "usage": {"prompt_tokens": 756, "completion_tokens": 132, "total_tokens": 888},
}


class _StubServer:
    """A local stand-in for the vLLM endpoint. Counts what it was asked."""

    def __init__(self, reply):
        import http.server
        import threading
        self.requests = []
        reply_bytes = json.dumps(reply).encode()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                outer.requests.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(reply_bytes)))
                self.end_headers()
                self.wfile.write(reply_bytes)

            def log_message(self, *args):
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _run_smoke_against(stub, tmp_path, cfg):
    import os
    import subprocess
    import sys
    revision = cfg.raw["models"]["generator"]["model"]["revision"]
    root = pathlib.Path(__file__).resolve().parents[1]
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3-14B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "model-00001-of-00001.safetensors"):
        (snapshot / name).write_text("{}")
    (snapshot.parents[1] / "refs").mkdir(parents=True)
    (snapshot.parents[1] / "refs" / "main").write_text(revision)
    runtime = tmp_path / "server_runtime.json"
    runtime.write_text(json.dumps({
        "host": "chomusuke02", "repo_id": "Qwen/Qwen3-14B", "revision": revision,
        "gpu_index": "0", "gpu_name": "NVIDIA RTX A6000", "dtype": "bfloat16",
        "max_model_len": 8192, "seed": 20260914, "generation_config": "vllm",
        "libraries": {"vllm": "0.8.5.post1+cu118", "transformers": "4.51.3",
                      "torch": "2.6.0+cu118"}}))
    result = subprocess.run(
        [sys.executable, "scripts/smoke_test.py", "--config", "configs/experiment.yaml",
         "--out", str(tmp_path / "smoke"), "--server-runtime", str(runtime),
         "--base-url", stub.base_url, "--send"],
        cwd=root, capture_output=True, text=True,
        env={**os.environ, "HF_HOME": str(tmp_path),
             "REASONSTYLE_ALLOW_LOCAL_GENERATION": "1", "HF_HUB_OFFLINE": "1"})
    return result


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    from reasonstyle.config import load_config
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "experiment.yaml")
    tmp_path = tmp_path_factory.mktemp("smoke_provenance")
    with _StubServer(SERVER_REPLY) as stub:
        result = _run_smoke_against(stub, tmp_path, cfg)
    return result, tmp_path / "smoke", stub


def test_a_validation_failure_is_not_reported_as_an_accepted_result(smoke_run):
    """The defect this fixes: the log said status ok while the validator had
    produced errors."""
    result, out, _ = smoke_run
    assert result.returncode == 1, result.stdout + result.stderr
    entries = [json.loads(line) for line in (out / "generation_log.jsonl").read_text().splitlines()]
    assert len(entries) == 1, "exactly one append-only entry per call"
    assert entries[0]["status"] == "validation_failed"
    assert entries[0]["stop_reason"] == "stop"
    assert "FAILED validation" in result.stderr


def test_the_saved_record_carries_the_validator_codes(smoke_run):
    _, out, _ = smoke_run
    entry = json.loads((out / "generation_log.jsonl").read_text().splitlines()[0])
    group = entry["validation"]["generated_group"]
    assert {"E_OPENING_REPEATED_IN_BODY", "E_PAIR_CONTENT_DRIFT",
            "E_WORD_RATIO_BODY", "E_WORD_RATIO_FULL_TEXT"} <= set(group["error_codes"])
    assert "W_PREMISE_NOT_IN_SCENARIO" in group["warning_codes"]
    assert entry["validation"]["ok"] is False


def test_the_validator_result_is_persisted_beside_the_raw_response(smoke_run):
    """A result that lives only in a terminal cannot be cited later."""
    _, out, _ = smoke_run
    entry = json.loads((out / "generation_log.jsonl").read_text().splitlines()[0])
    saved = pathlib.Path(entry["validation"]["report_file"])
    assert saved.is_file()
    report = json.loads(saved.read_text())
    # The saved summary, the log entry and the file on disk must all name the
    # same path: the summary used to be serialised before the path was known,
    # so the file said "report_file": null.
    assert report["summary"]["report_file"] is not None
    assert (pathlib.Path(report["summary"]["report_file"]).resolve()
            == saved.resolve() == (out / f"validation_{entry['call_id']}.json").resolve())
    assert report["call_id"] == entry["call_id"]
    assert report["report"]["ok"] is False
    assert {f["code"] for f in report["report"]["findings"]} >= {"E_PAIR_CONTENT_DRIFT"}
    assert (out / "raw" / f"{entry['call_id']}.json").is_file()


def test_human_review_counts_are_labelled_by_scope(smoke_run):
    result, out, _ = smoke_run
    entry = json.loads((out / "generation_log.jsonl").read_text().splitlines()[0])
    group = entry["validation"]["generated_group"]["human_review_outstanding"]
    whole = entry["validation"]["whole_fixture_corpus"]["human_review_outstanding"]
    assert 0 < group < whole, "the generated group is a part of the fixture corpus"
    assert "generated group" in result.stdout and "whole fixture corpus" in result.stdout


def test_the_stop_reason_is_shown_and_recorded(smoke_run):
    result, out, _ = smoke_run
    assert re.search(r"stop_reason\s+stop", result.stdout)
    entry = json.loads((out / "generation_log.jsonl").read_text().splitlines()[0])
    assert entry["stop_reason"] == "stop"
    assert entry["usage"]["completion_tokens"] == 132


def test_exactly_one_call_with_no_repair_and_no_retry(smoke_run):
    result, out, stub = smoke_run
    assert len(stub.requests) == 1
    assert "No repair request was built" in result.stderr
    assert not list(out.glob("*repair*"))


def test_the_pinned_bodies_are_the_ones_the_model_returned():
    """Guards against these strings drifting from the evidence. The evidence is
    gitignored, so this skips where it is not present."""
    if not EVIDENCE.is_file():
        pytest.skip(f"{EVIDENCE} is not on this machine (gitignored evidence)")
    raw = json.loads(EVIDENCE.read_text())
    choice = raw["response"]["choices"][0]
    assert json.loads(choice["message"]["content"]) == SMOKE_BODIES
    assert choice["finish_reason"] == "stop"
