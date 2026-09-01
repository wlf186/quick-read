from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_exit(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.05)
    return not process_exists(pid)


def scaffold_project(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    project = tmp_path / "project"
    scripts = project / "scripts"
    bin_dir = project / ".venv" / "bin"
    package = project / "fake-src" / "sandevistan_read"
    scripts.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    package.mkdir(parents=True)

    shutil.copy2(REPOSITORY_ROOT / "scripts" / "start.sh", scripts / "start.sh")
    shutil.copy2(REPOSITORY_ROOT / "scripts" / "stop.sh", scripts / "stop.sh")
    (bin_dir / "python").symlink_to(sys.executable)

    executable = bin_dir / "sandevistan-read"
    executable.write_text(
        f"""#!{sys.executable}
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if os.environ.get("TEST_SERVER_EXIT") == "1":
    raise SystemExit(3)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/auth/status":
            self.send_error(404)
            return
        body = b'{{"required": false, "authenticated": true}}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

ThreadingHTTPServer(("127.0.0.1", int(os.environ["TEST_SERVER_PORT"])), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "config.py").write_text(
        """import os

class Server:
    host = "127.0.0.1"
    port = int(os.environ["TEST_SERVER_PORT"])

class Config:
    server = Server()

CONFIG = Config()
""",
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "fake-src")
    environment["TEST_SERVER_PORT"] = str(available_port())
    return project, environment


def run_script(project: Path, name: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(project / "scripts" / name)],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def force_cleanup(project: Path) -> None:
    pid_file = project / "runtime" / "run" / "server.pid"
    if not pid_file.exists():
        return
    value = pid_file.read_text(encoding="utf-8").strip()
    if value.isdigit() and process_exists(int(value)):
        os.kill(int(value), signal.SIGKILL)


def test_start_detaches_is_idempotent_and_stops(tmp_path: Path) -> None:
    project, environment = scaffold_project(tmp_path)
    pid_file = project / "runtime" / "run" / "server.pid"
    try:
        started = run_script(project, "start.sh", environment)
        assert started.returncode == 0, started.stderr
        assert "Sandevistan-Read started" in started.stdout
        pid = int(pid_file.read_text(encoding="utf-8"))
        assert process_exists(pid)

        with urllib.request.urlopen(
            f"http://127.0.0.1:{environment['TEST_SERVER_PORT']}/auth/status", timeout=2
        ) as response:
            assert response.status == 200

        repeated = run_script(project, "start.sh", environment)
        assert repeated.returncode == 0, repeated.stderr
        assert f"already running (PID {pid})" in repeated.stdout
        assert int(pid_file.read_text(encoding="utf-8")) == pid

        stopped = run_script(project, "stop.sh", environment)
        assert stopped.returncode == 0, stopped.stderr
        assert "Sandevistan-Read stopped" in stopped.stdout
        assert wait_for_exit(pid)
        assert not pid_file.exists()
    finally:
        force_cleanup(project)


def test_stop_rejects_pid_owned_by_another_process(tmp_path: Path) -> None:
    project, environment = scaffold_project(tmp_path)
    pid_file = project / "runtime" / "run" / "server.pid"
    pid_file.parent.mkdir(parents=True)
    unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)
    pid_file.write_text(f"{unrelated.pid}\n", encoding="utf-8")
    try:
        stopped = run_script(project, "stop.sh", environment)
        assert stopped.returncode == 1
        assert "belongs to another process" in stopped.stderr
        assert unrelated.poll() is None
        assert not pid_file.exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


@pytest.mark.parametrize("pid_value", ["not-a-pid", "999999999"])
def test_stop_cleans_invalid_or_stale_pid_file(tmp_path: Path, pid_value: str) -> None:
    project, environment = scaffold_project(tmp_path)
    pid_file = project / "runtime" / "run" / "server.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(f"{pid_value}\n", encoding="utf-8")

    stopped = run_script(project, "stop.sh", environment)

    assert not pid_file.exists()
    if pid_value.isdigit():
        assert stopped.returncode == 0
        assert "stale PID file" in stopped.stdout
    else:
        assert stopped.returncode == 1
        assert "invalid PID file" in stopped.stderr


def test_failed_start_removes_pid_file(tmp_path: Path) -> None:
    project, environment = scaffold_project(tmp_path)
    environment["TEST_SERVER_EXIT"] = "1"

    started = run_script(project, "start.sh", environment)

    assert started.returncode == 1
    assert "failed to start" in started.stderr
    assert not (project / "runtime" / "run" / "server.pid").exists()
