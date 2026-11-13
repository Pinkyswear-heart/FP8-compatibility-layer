"""Runs tests/bench_transcode.py and turns its JSON into a record.

The benchmark is invoked as a subprocess rather than imported, so a crash or a mistyped flag
cannot take the page down with it, and the run keeps going if the browser is closed. Its stdout
is streamed into the job log -- a full run is about a minute, and half of this project's
conclusions are about watching which section got slow.

    from benchlab.runner import Job
    job = Job(root=records_dir, repo=repo_root, only=["bw", "decode"])
    job.start(); ...; job.snapshot()
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time

from . import store


class Job:
    """One benchmark run in a background thread. At most one job per process."""

    def __init__(self, root: str, repo: str, only: list[str] | None = None,
                 note: str = "", python: str | None = None):
        self.root = root
        self.repo = repo
        self.only = only or None
        self.note = note
        self.python = python or sys.executable
        self.status = "idle"          # idle | running | done | failed
        self.log = ""
        self.record_id = None
        self.error = None
        self.started = None
        self.ended = None
        self._proc = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ control
    def start(self) -> "Job":
        with self._lock:
            if self.status == "running":
                raise RuntimeError("a benchmark is already running")
            self.status = "running"
            self.started = time.time()
            self.log = ""
            self.error = None
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        script = os.path.join(self.repo, "tests", "bench_transcode.py")
        if not os.path.isfile(script):
            self._finish("failed", error=f"benchmark not found: {script}")
            return

        fd, out_path = tempfile.mkstemp(prefix="benchlab_", suffix=".json")
        os.close(fd)
        # -u: -I implies -E, which drops PYTHONUNBUFFERED, so without this the log arrives in one
        # block when the run ends and the page shows a dead "运行中" for the whole minute.
        cmd = [self.python, "-I", "-u", script, "--json", out_path]
        if self.only:
            cmd += ["--only", ",".join(self.only)]

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=self.repo, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1,
            )
            for line in self._proc.stdout:
                with self._lock:
                    self.log += line
                    if len(self.log) > 400_000:
                        self.log = self.log[-300_000:]
            rc = self._proc.wait()
        except Exception as e:  # noqa: BLE001
            self._finish("failed", error=repr(e))
            return

        if rc != 0:
            self._finish("failed", error=f"benchmark exited {rc}; see the log")
            return

        try:
            with open(out_path, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:  # noqa: BLE001
            self._finish("failed", error=f"could not read the benchmark json: {e!r}")
            return
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

        try:
            rec = store.record_from_raw(raw, source="local", note=self.note)
            store.save(rec, self.root)
        except Exception as e:  # noqa: BLE001
            self._finish("failed", error=f"could not build the record: {e!r}")
            return

        with self._lock:
            self.record_id = rec["id"]
        self._finish("done")

    def _finish(self, status: str, error: str | None = None):
        with self._lock:
            self.status = status
            self.error = error
            self.ended = time.time()

    def cancel(self):
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass

    # ------------------------------------------------------------------ view
    def snapshot(self, tail: int = 200) -> dict:
        with self._lock:
            lines = self.log.splitlines()
            return {
                "status": self.status,
                "error": self.error,
                "record_id": self.record_id,
                "elapsed_s": round((self.ended or time.time()) - self.started, 1)
                if self.started else 0,
                "log_tail": "\n".join(lines[-tail:]),
            }
