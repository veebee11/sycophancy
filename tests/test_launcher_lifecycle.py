"""The launcher's runtime-record lifecycle, executed with a fake child.

`scripts/server/serve_vllm.sh` is sourced, which defines its lifecycle functions
and stops before any launch logic, so these tests run the launcher's real
`prepare_launch` and `supervise` with a small Python HTTP server standing in for
vLLM. Nothing here touches a GPU, a model or any host other than 127.0.0.1.

The property under test: `server_runtime.json` exists only while this
launcher's own child is alive and answering /health. A record surviving a failed
start would look valid to the smoke test.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

import pytest

LAUNCHER = pathlib.Path(__file__).resolve().parents[1] / "scripts/server/serve_vllm.sh"

FAKE_SERVER = """
import os, sys, time, http.server
port, delay, lifetime = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if len(sys.argv) > 4:
    open(sys.argv[4], "w").write(str(os.getpid()))
time.sleep(delay)
class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass
server = http.server.HTTPServer(("127.0.0.1", port), Health)
server.timeout = 0.1
end = time.monotonic() + lifetime
while time.monotonic() < end:
    server.handle_request()
"""

RECORD = {"revision": "a" * 40, "generation_config": "vllm", "gpu_index": "0"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _responds(port: int) -> bool:
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        opener.open(f"http://127.0.0.1:{port}/health", timeout=1).close()
        return True
    except OSError:
        return False


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class Harness:
    def __init__(self, tmp_path: pathlib.Path):
        self.tmp = tmp_path
        self.port = _free_port()
        self.record = tmp_path / "pilot" / "server_runtime.json"
        self.pending = self.record.with_name(self.record.name + ".pending")
        self.record.parent.mkdir(parents=True)
        self.fake = tmp_path / "fake_server.py"
        self.fake.write_text(FAKE_SERVER)
        self.marker = tmp_path / "child_started"

    def child(self, delay: float = 0.0, lifetime: float = 60.0, pid_file: bool = True) -> str:
        """The command line of a fake vLLM: healthy after `delay`, gone after `lifetime`."""
        args = [sys.executable, str(self.fake), str(self.port), str(delay), str(lifetime)]
        if pid_file:
            args.append(str(self.marker))
        return " ".join(f"'{a}'" for a in args)

    def script(self, body: str) -> str:
        return (f"set -euo pipefail\nsource '{LAUNCHER}'\n"
                f"PYTHON='{sys.executable}'\nPORT={self.port}\n"
                f"RUNTIME_RECORD='{self.record}'\nCHILD=''\n"
                f"HEALTH_POLL_SECONDS=0.1\n{body}\n")

    def write_pending(self) -> str:
        """Shell that writes the pending record, as the launcher does, and notes its inode."""
        return (f"printf '%s\\n' '{json.dumps(RECORD)}' > '{self.pending}'\n"
                f"'{sys.executable}' -c 'import os,sys; print(os.stat(sys.argv[1]).st_ino)' "
                f"'{self.pending}' > '{self.tmp / 'pending_inode'}'\n")

    def run(self, body: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", self.script(body)], capture_output=True,
                              text=True, timeout=timeout)

    def start(self, body: str) -> subprocess.Popen:
        return subprocess.Popen(["bash", "-c", self.script(body)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def child_pid(self) -> int | None:
        return int(self.marker.read_text()) if self.marker.exists() else None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    pid = harness.child_pid()
    if pid and _alive(pid):
        os.kill(pid, signal.SIGKILL)


def test_a_stale_active_and_pending_record_are_removed_before_launch(h):
    h.record.write_text(json.dumps(RECORD))
    h.pending.write_text(json.dumps(RECORD))
    result = h.run(f"prepare_launch\n"
                   f"[[ ! -e '{h.record}' && ! -e '{h.pending}' ]] && echo removed-before-launch\n"
                   f"trap - EXIT")
    assert result.returncode == 0, result.stderr
    assert "removed-before-launch" in result.stdout
    assert not h.marker.exists()


def test_a_port_that_already_responds_is_refused_before_any_child(h):
    """The host is shared: another service on the port is never recorded as ours,
    and the record on disk (which may describe that running server) is left alone."""
    other = subprocess.Popen([sys.executable, str(h.fake), str(h.port), "0", "60"])
    try:
        assert _wait_for(lambda: _responds(h.port))
        h.record.write_text(json.dumps(RECORD))
        result = h.run(f"prepare_launch\n{h.write_pending()}supervise {h.child()}")
    finally:
        other.kill()
        other.wait()
    assert result.returncode != 0
    assert "already responds" in result.stderr
    assert not h.marker.exists(), "no child may be started"
    assert not h.pending.exists()
    assert json.loads(h.record.read_text()) == RECORD


def test_a_child_that_exits_before_becoming_healthy_never_publishes_a_record(h):
    failing = f"'{sys.executable}' -c 'import sys; sys.exit(3)'"
    result = h.run(f"prepare_launch\n{h.write_pending()}supervise {failing}")
    assert result.returncode != 0
    assert "exited (status 3) before becoming healthy" in result.stderr
    assert not h.record.exists()
    assert not h.pending.exists()


def test_a_child_that_never_becomes_healthy_is_stopped_at_the_timeout(h):
    result = h.run(f"HEALTH_TIMEOUT_SECONDS=1\nprepare_launch\n{h.write_pending()}"
                   f"supervise {h.child(delay=30)}")
    assert result.returncode != 0
    assert "not healthy within 1s" in result.stderr
    assert not h.record.exists() and not h.pending.exists()
    pid = h.child_pid()
    assert pid and _wait_for(lambda: not _alive(pid), 10), "the child must be stopped"


def test_a_healthy_child_atomically_promotes_the_pending_record(h):
    proc = h.start(f"prepare_launch\n{h.write_pending()}supervise {h.child(delay=1.0)}")
    try:
        # Not before the child answers /health ...
        assert _wait_for(h.marker.exists)
        assert not h.record.exists()
        # ... and then by rename: same inode, identical content, no pending left.
        assert _wait_for(h.record.exists), proc.stderr.read() if proc.poll() is not None else ""
        assert _responds(h.port)
        assert not h.pending.exists()
        assert json.loads(h.record.read_text()) == RECORD
        inode = int((h.tmp / "pending_inode").read_text())
        assert h.record.stat().st_ino == inode
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)


def test_terminating_the_launcher_stops_the_child_and_removes_the_record(h):
    proc = h.start(f"prepare_launch\n{h.write_pending()}supervise {h.child()}")
    assert _wait_for(h.record.exists)
    pid = h.child_pid()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=20) != 0
    assert pid and _wait_for(lambda: not _alive(pid), 10), "termination must reach the child"
    assert not h.record.exists() and not h.pending.exists()
    assert not _responds(h.port)


def test_the_record_is_removed_when_the_engine_exits_after_becoming_healthy(h):
    proc = h.start(f"prepare_launch\n{h.write_pending()}supervise {h.child(lifetime=2.0)}")
    assert _wait_for(h.record.exists)
    proc.wait(timeout=30)
    assert not h.record.exists() and not h.pending.exists()


def test_the_launch_path_uses_the_lifecycle_in_order():
    """The executed part of the launcher: refuse or clean up first, write only a
    pending record, and launch through supervise rather than exec."""
    lines = [line for line in LAUNCHER.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    body = lines[lines.index('[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0'):]
    text = "\n".join(body)
    prepare = text.index("\nprepare_launch")
    preflight = text.index("preflight_model.py")
    pending = text.index('"$RUNTIME_RECORD.pending" <<PY')
    launch = text.index('supervise env CUDA_VISIBLE_DEVICES="$GPU" "$VLLM" serve')
    assert prepare < preflight < pending < launch
    assert "exec " not in text
    assert 'open(sys.argv[1], "w")' in text and '"$PYTHON" - "$RUNTIME_RECORD" <<PY' not in text
