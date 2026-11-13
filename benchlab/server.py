"""Local web app: run the FP8 transcode benchmark, keep one record per GPU, compare any two.

    python -m benchlab                 # http://127.0.0.1:8770
    python -m benchlab --port 8771 --open

Deliberately stdlib-only. It shells out to tests/bench_transcode.py, which owns every
measurement; this file owns storage, the comparison arithmetic and the page. That split is what
lets a record made on another card be dropped in as JSON and compared here, on a machine with no
GPU at all.

Binds 127.0.0.1 by default: the run endpoint starts arbitrary benchmark subprocesses.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import webbrowser
from http import server as http_server
from urllib.parse import parse_qs, urlparse

from . import store
from .runner import Job

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
REPO = os.path.dirname(HERE)
DEFAULT_ROOT = os.path.join(REPO, "benchlab", "records")

_ID_RE = re.compile(r"^[a-z0-9-]+/\d{8}T\d{6}$")


class State:
    def __init__(self, root: str):
        self.root = root
        self.job: Job | None = None
        os.makedirs(root, exist_ok=True)

    def current_job(self) -> Job | None:
        return self.job

    def start(self, only, note) -> Job:
        if self.job is not None and self.job.status == "running":
            raise RuntimeError("a benchmark is already running")
        self.job = Job(root=self.root, repo=REPO, only=only, note=note)
        self.job.start()
        return self.job


def _json(handler, obj, code=200):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler) -> dict:
    n = int(handler.headers.get("Content-Length") or 0)
    if n <= 0:
        return {}
    if n > 64 << 20:
        raise ValueError("body too large")
    return json.loads(handler.rfile.read(n).decode("utf-8"))


def _public(rec: dict) -> dict:
    """Record without the raw harness JSON, which can be hundreds of KB per card."""
    return {k: v for k, v in rec.items() if k != "raw" and k != "_path"}


class Handler(http_server.BaseHTTPRequestHandler):
    state: State = None  # set in main()

    # ------------------------------------------------------------------ routing
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/meta":
                return _json(self, {
                    "metrics": [
                        {"key": k, "label": l, "unit": u_, "higher_is_better": h,
                         "group": g, "hint": hint}
                        for (k, l, u_, h, g, hint) in store.METRICS
                    ],
                    "groups": [{"key": k, "label": l} for k, l in store.GROUPS],
                    "root": self.state.root,
                    "repo": REPO,
                })
            if path == "/api/records":
                recs = store.load_all(self.state.root)
                return _json(self, {"records": [_public(r) for r in recs]})
            if path.startswith("/api/records/"):
                rid = path[len("/api/records/"):].rstrip("/")
                if not _ID_RE.match(rid):
                    return _json(self, {"error": "bad id"}, 400)
                for r in store.load_all(self.state.root):
                    if r["id"] == rid:
                        return _json(self, _public(r))
                return _json(self, {"error": "not found"}, 404)
            if path == "/api/job":
                job = self.state.current_job()
                return _json(self, job.snapshot() if job else {"status": "idle", "log_tail": ""})
            return _json(self, {"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            return _json(self, {"error": repr(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/run":
                body = _read_json(self)
                only = body.get("only") or None
                if only is not None:
                    only = [s for s in only if s in
                            {"bw", "decode", "encode", "gemv", "gemm", "floor"}]
                    if not only:
                        return _json(self, {"error": "no valid sections"}, 400)
                job = self.state.start(only, body.get("note") or "")
                return _json(self, job.snapshot())
            if u.path == "/api/job/cancel":
                job = self.state.current_job()
                if job:
                    job.cancel()
                return _json(self, job.snapshot() if job else {"status": "idle"})
            if u.path == "/api/import":
                body = _read_json(self)
                raw = body.get("raw") or body
                if not isinstance(raw, dict) or "sections" not in raw:
                    return _json(self, {"error": "expected a tests/bench_transcode.py json"}, 400)
                rec = store.record_from_raw(raw, source="import", note=body.get("note") or "")
                if any(r["id"] == rec["id"] for r in store.load_all(self.state.root)):
                    rec["id"] += "-1"
                store.save(rec, self.state.root)
                return _json(self, _public(rec))
            if u.path == "/api/compare":
                body = _read_json(self)
                ids = body.get("ids") or []
                recs = {r["id"]: r for r in store.load_all(self.state.root)}
                chosen = [recs[i] for i in ids if i in recs]
                return _json(self, store.compare(chosen))
            return _json(self, {"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            return _json(self, {"error": repr(e)}, 500)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/api/records/"):
            return _json(self, {"error": "not found"}, 404)
        rid = path[len("/api/records/"):].rstrip("/")
        if not _ID_RE.match(rid):
            return _json(self, {"error": "bad id"}, 400)
        ok = store.delete(rid, self.state.root)
        return _json(self, {"ok": ok}, 200 if ok else 404)

    # ------------------------------------------------------------------ static
    def _static(self, rel: str):
        rel = rel.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC, rel))
        if not full.startswith(os.path.normpath(STATIC)) or not os.path.isfile(full):
            return _json(self, {"error": "not found"}, 404)
        ctype = {".html": "text/html; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".svg": "image/svg+xml"}.get(os.path.splitext(full)[1], "application/octet-stream")
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if os.environ.get("BENCHLAB_VERBOSE"):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchlab")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="where records are stored")
    ap.add_argument("--open", action="store_true", help="open the page in a browser")
    a = ap.parse_args(argv)

    Handler.state = State(a.root)
    srv = http_server.ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}/"
    print(f"benchlab on {url}   records: {a.root}")
    print("  Ctrl-C to stop")
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
