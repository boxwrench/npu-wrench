# SPDX-FileCopyrightText: Copyright (C) 2026 boxwrench
# SPDX-License-Identifier: Apache-2.0
"""A background memory-writer server for Hy-Memory on the NPU (standard library only).

Built for the job rather than for chat:
  * idle-gated scheduling: writes run only after the iGPU (GPU busy % from sysfs) has been idle for
    --idle-hold seconds, and pause at the next prefill layer or decode token when it becomes busy, so a
    foreground model on the iGPU runs undisturbed; paused work resumes, it is not thrown away;
  * an async write queue (priority, then arrival) with per-session watermarks: ``received_through`` (highest
    span end submitted) and ``extracted_through`` (highest span end such that every submitted span up to it is
    done, so a correction finishing early never advances it past unfinished work). ``compacted_through``
    (reconciled into durable memory) belongs to the memory system, not to the writer;
  * an overload budget: under a foreground that never idles, a write still gets ``--min-duty`` of the time,
    so the backlog stays bounded;
  * a prompt-prefix cache: the KV cache of the previous request is reused for the longest shared prefix;
  * an OpenAI-compatible ``/v1/chat/completions`` (synchronous, top priority) for drop-in use.

  python -m npuwrench.server --weights DIR [--port 8099] [--prefill 4096] [--yield-busy 15] [--min-duty 0.1]

API
  POST /v1/memory/write        {"messages": [...], "session": "s1", "span_end": 12, "priority": 5,
                                "max_new_tokens": 512}               -> 202 {"id": ...}
  GET  /v1/memory/write/<id>   job status, result text, valid_json, timings, seconds deferred
  GET  /v1/memory/watermark/<session>         -> {"received_through": ..., "extracted_through": ...}
  POST /v1/chat/completions    OpenAI chat format (greedy)
  GET  /v1/models, /health, /metrics
"""
import argparse
import glob
import heapq
import itertools
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Foreground:
    """Smoothed iGPU busy percentage from amdgpu's sysfs counter, and how long the foreground has been idle."""

    def __init__(self, threshold=15.0, idle_hold=2.0, period=0.05, alpha=0.2):
        paths = glob.glob("/sys/class/drm/card*/device/gpu_busy_percent")
        self.path = paths[0] if paths else None
        self.threshold, self.idle_hold, self.period, self.alpha = threshold, idle_hold, period, alpha
        self.ema, self.last_busy = 0.0, 0.0
        if self.path and threshold > 0:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                v = float(open(self.path).read())
            except (OSError, ValueError):
                v = 0.0
            self.ema = self.alpha * v + (1 - self.alpha) * self.ema
            if self.ema > self.threshold:
                self.last_busy = time.time()
            time.sleep(self.period)

    def busy(self):
        """True while the foreground is busy or was busy within the last ``idle_hold`` seconds."""
        if self.path is None or self.threshold <= 0:
            return False
        return self.ema > self.threshold or time.time() - self.last_busy < self.idle_hold


class Job:
    def __init__(self, messages, session=None, span_end=None, priority=5, max_new=512, sync=False):
        self.id = uuid.uuid4().hex[:12]
        self.messages, self.session, self.span_end = messages, session, span_end
        self.priority, self.max_new, self.sync = priority, max_new, sync
        self.status, self.text, self.valid_json, self.error = "queued", None, None, None
        self.created, self.started, self.finished = time.time(), None, None
        self.deferred_s, self.prompt_tokens, self.generated, self.reused = 0.0, 0, 0, 0
        self.prompt_s, self.decode_ms = None, None
        self.done = threading.Event()

    def view(self):
        return {k: getattr(self, k) for k in ("id", "status", "session", "span_end", "priority", "text", "valid_json",
                                              "error", "created", "started", "finished", "deferred_s", "prompt_tokens",
                                              "generated", "reused", "prompt_s", "decode_ms")}


class Writer:
    def __init__(self, weights, prefill=4096, max_ctx=None, experts_per_dispatch=3, yield_busy=15.0, min_duty=0.1,
                 idle_hold=2.0):
        import jinja2
        from tokenizers import Tokenizer
        from .engine import HyMemoryNPU
        self.tok = Tokenizer.from_file(os.path.join(weights, "tokenizer.json"))
        self.tpl = jinja2.Environment().from_string(open(os.path.join(weights, "chat_template.jinja")).read())
        eos = self.tok.token_to_id("<eos:6124c78e>")
        self.stop = [eos] if eos is not None else []
        self.max_ctx = max_ctx or prefill + 1024
        self.eng = HyMemoryNPU(weights, max_ctx=self.max_ctx, experts_per_dispatch=experts_per_dispatch,
                               prefill_tokens=prefill)
        self.fg, self.min_duty = Foreground(yield_busy, idle_hold), min_duty
        self.q, self.seq, self.lock, self.cv = [], itertools.count(), threading.Lock(), threading.Condition()
        self.jobs, self.cached = {}, []
        self.spans = {}  # session -> {"submitted": {span_end: done?}}
        self.stats = {"writes": 0, "deferred_s": 0.0, "reused_tokens": 0, "prompt_tokens": 0}
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, job):
        if job.session is not None and job.span_end is not None:
            with self.lock:
                self.spans.setdefault(job.session, {})[job.span_end] = False
        with self.cv:
            self.jobs[job.id] = job
            heapq.heappush(self.q, (job.priority, next(self.seq), job))
            self.cv.notify()
        return job

    def watermarks(self, session):
        with self.lock:
            d = self.spans.get(session, {})
            received = max(d) if d else None
            extracted = None
            for end in sorted(d):           # contiguous: stop at the first span not done yet
                if not d[end]:
                    break
                extracted = end
        return {"session": session, "received_through": received, "extracted_through": extracted}

    def _hook_for(self, job):
        """Yield at safe boundaries (between prefill layers and decode tokens) while the foreground is busy, but
        keep at least ``min_duty`` of wall time for the write so a backlog cannot grow without bound."""
        state = {"run": 0.0, "mark": time.time()}

        def hook():
            if job.sync:
                return  # interactive calls never yield
            now = time.time()
            state["run"] += now - state["mark"]
            t0 = now
            while self.fg.busy():
                waited = job.deferred_s + (time.time() - t0)
                if self.min_duty > 0 and state["run"] < self.min_duty * (state["run"] + waited):
                    break               # owed duty: take one step even though the foreground is busy
                time.sleep(0.05)
            job.deferred_s += time.time() - t0
            state["mark"] = time.time()
        return hook

    def _worker(self):
        while True:
            with self.cv:
                while not self.q:
                    self.cv.wait()
                _, _, job = heapq.heappop(self.q)
            job.status, job.started = "running", time.time()
            try:
                prompt = self.tpl.render(messages=job.messages, add_generation_prompt=True)
                ids = self.tok.encode(prompt, add_special_tokens=False).ids
                if len(ids) + job.max_new > self.max_ctx:
                    raise ValueError(f"prompt {len(ids)} + {job.max_new} tokens exceeds max context {self.max_ctx}")
                lcp = 0
                for a, b in zip(self.cached, ids):
                    if a != b:
                        break
                    lcp += 1
                out, tp, per = self.eng.generate(ids, job.max_new, stop_ids=self.stop, start=lcp,
                                                 hook=self._hook_for(job))
                # positions 0 .. len(ids) + len(out) - 2 are now cached (the last token was never fed back)
                self.cached = ids + out[:-1]
                gen = [t for t in out if t not in self.stop]
                job.text = self.tok.decode(gen)
                try:
                    json.loads(job.text.strip())
                    job.valid_json = True
                except ValueError:
                    job.valid_json = False
                job.prompt_tokens, job.generated, job.reused = len(ids), len(out), min(lcp, len(ids) - 1)
                job.prompt_s = round(tp, 3)
                job.decode_ms = round(1e3 * sorted(per)[len(per) // 2], 1) if per else None
                job.status = "done"
                if job.session is not None and job.span_end is not None:
                    with self.lock:
                        self.spans[job.session][job.span_end] = True
                self.stats["writes"] += 1
                self.stats["deferred_s"] += job.deferred_s
                self.stats["reused_tokens"] += job.reused
                self.stats["prompt_tokens"] += job.prompt_tokens
            except Exception as ex:  # report, keep serving
                job.status, job.error = "error", f"{type(ex).__name__}: {ex}"
                self.cached = []
            job.finished = time.time()
            job.done.set()


def make_handler(w):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/health":
                return self._send(200, {"status": "ok"})
            if p == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": "hy-memory-30b-a3b-npu", "object": "model"}]})
            if p == "/metrics":
                return self._send(200, {**w.stats, "queued": len(w.q), "foreground_busy_pct": round(w.fg.ema, 1),
                                        "yielding": w.fg.busy()})
            if p.startswith("/v1/memory/write/"):
                j = w.jobs.get(p.rsplit("/", 1)[1])
                return self._send(200, j.view()) if j else self._send(404, {"error": "no such job"})
            if p.startswith("/v1/memory/watermark/"):
                return self._send(200, w.watermarks(p.rsplit("/", 1)[1]))
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.rstrip("/")
            try:
                b = self._body()
            except ValueError:
                return self._send(400, {"error": "bad json"})
            if p == "/v1/memory/write":
                j = w.submit(Job(b["messages"], b.get("session"), b.get("span_end"), int(b.get("priority", 5)),
                                 int(b.get("max_new_tokens", 512))))
                return self._send(202, {"id": j.id})
            if p == "/v1/chat/completions":
                j = w.submit(Job(b["messages"], priority=0, max_new=int(b.get("max_tokens", 512)), sync=True))
                j.done.wait()
                if j.status != "done":
                    return self._send(500, {"error": j.error})
                return self._send(200, {
                    "id": "chatcmpl-" + j.id, "object": "chat.completion", "created": int(j.created),
                    "model": "hy-memory-30b-a3b-npu",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": j.text},
                                 "finish_reason": "length" if j.generated >= j.max_new else "stop"}],
                    "usage": {"prompt_tokens": j.prompt_tokens, "completion_tokens": j.generated,
                              "total_tokens": j.prompt_tokens + j.generated}})
            return self._send(404, {"error": "not found"})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--prefill", type=int, default=4096, help="max prompt tokens for NPU prefill")
    ap.add_argument("--experts-per-dispatch", type=int, default=3, choices=(1, 3))
    ap.add_argument("--yield-busy", type=float, default=15.0,
                    help="pause background writes while the iGPU is busier than this %% (0 = never yield)")
    ap.add_argument("--idle-hold", type=float, default=2.0,
                    help="write only after the foreground has been idle this many seconds")
    ap.add_argument("--min-duty", type=float, default=0.1,
                    help="share of wall time a write keeps even while the foreground is busy (bounds the backlog)")
    a = ap.parse_args()
    w = Writer(a.weights, a.prefill, experts_per_dispatch=a.experts_per_dispatch, yield_busy=a.yield_busy,
               min_duty=a.min_duty, idle_hold=a.idle_hold)
    print(f"npuwrench server on http://{a.host}:{a.port} (yield above {a.yield_busy}% iGPU busy)", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(w)).serve_forever()


if __name__ == "__main__":
    main()
