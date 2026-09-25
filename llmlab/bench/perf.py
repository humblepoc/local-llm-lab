"""Performance benchmarking.

Measures what actually matters on a memory-bound CPU box:

  * load time      — spawn to /health (CPU load of the whole weight tensor)
  * TTFT           — time to first streamed token
  * prefill tok/s  — prompt processing rate
  * decode tok/s   — generation rate
  * peak RSS       — resident memory of the server process

Run at several context sizes because the KV cache, not the weights, is what
pushes a 32GB machine over the edge.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..manifest import Model
from ..runtimes.base import Runtime, ServerHandle

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "def parse_token(s: str) -> Token: return Token(s.strip().lower())\n"
)

# llama.cpp recognises these as reasoning fields. A reasoning model streams
# into one of them rather than `content`, so counting only `content` reports
# 0 tok/s for exactly the models this framework is built to evaluate.
TEXT_FIELDS = (
    "content",
    "reasoning_content",
    "reasoning",
    "reasoning_text",
    "thinking",
)


def _delta_text(delta: dict) -> str:
    """Concatenate any text-bearing field from a streaming delta."""
    if not isinstance(delta, dict):
        return ""
    out: list[str] = []
    for field in TEXT_FIELDS:
        val = delta.get(field)
        if isinstance(val, str) and val:
            out.append(val)
    return "".join(out)


@dataclass
class PerfSample:
    ctx_size: int
    load_seconds: float
    ttft_seconds: float
    decode_tps: float
    prefill_tps: float | None
    peak_rss_gb: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


@dataclass
class PerfReport:
    model_id: str
    name: str
    arch: str
    params: str
    active: str | None
    runtime: str
    threads: int
    host: str
    samples: list[PerfSample] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    # -- rollups -----------------------------------------------------------
    @property
    def best_decode(self) -> float | None:
        vals = [s.decode_tps for s in self.samples if not s.error]
        return max(vals) if vals else None

    @property
    def mean_decode(self) -> float | None:
        vals = [s.decode_tps for s in self.samples if not s.error]
        return round(statistics.mean(vals), 2) if vals else None

    @property
    def max_peak_rss(self) -> float | None:
        vals = [s.peak_rss_gb for s in self.samples if not s.error]
        return round(max(vals), 2) if vals else None

    @property
    def mean_load(self) -> float | None:
        vals = [s.load_seconds for s in self.samples if not s.error]
        return round(statistics.mean(vals), 2) if vals else None


class _RssWatcher:
    """Samples a process's working set on a background thread."""

    def __init__(self, proc, interval: float = 0.25) -> None:
        self.proc = proc
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                # psutil is optional; fall back to a cheap check.
                try:
                    import psutil  # type: ignore

                    rss = psutil.Process(self.proc.pid).memory_info().rss
                    self.peak = max(self.peak, rss / 1_073_741_824)
                except ImportError:
                    # Windows fallback via ctypes is not worth the complexity;
                    # report 0 and let the caller note it.
                    pass
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def _rss_gb(proc) -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.Process(proc.pid).memory_info().rss / 1_073_741_824, 2)
    except Exception:
        return 0.0


def measure_once(
    runtime: Runtime,
    model: Model,
    port: int,
    ctx_size: int,
    max_tokens: int = 192,
) -> PerfSample:
    """Start a server at ``ctx_size``, measure, then shut it down."""
    handle: ServerHandle | None = None
    watcher = _RssWatcher(None)  # replaced below once the proc exists
    try:
        t0 = time.time()
        handle = runtime.start(model, port, ctx_size=ctx_size)
        watcher = _RssWatcher(handle.process)
        watcher.start()

        if not runtime.wait_ready(handle, timeout=900):
            err = "server did not become healthy in time"
            if runtime.__class__.__name__ == "LlamaCppRuntime":
                err += "\n" + runtime.tail_log(handle)
            return PerfSample(ctx_size, time.time() - t0, 0.0, 0.0, None, 0.0, 0, 0, err)

        load_s = time.time() - t0

        # --- generation: TTFT + decode rate ---------------------------
        # Count every text-bearing delta field, not just `content`:
        # reasoning models stream into a separate field and would otherwise
        # measure as 0 tok/s with a TTFT equal to total generation time.
        messages = [
            {"role": "user", "content": "Write a Python function that reverses a linked list. Code only."}
        ]
        first_token_at = None
        token_chunks = 0
        gen_start = time.time()
        for elapsed, chunk in runtime.chat_stream(
            handle, messages, max_tokens=max_tokens, temperature=0.0
        ):
            choices = chunk.get("choices") or []
            if choices and _delta_text(choices[0].get("delta") or {}):
                if first_token_at is None:
                    first_token_at = elapsed
                token_chunks += 1
        gen_total = time.time() - gen_start

        ttft = first_token_at if first_token_at is not None else gen_total
        decode_elapsed = max(gen_total - ttft, 1e-9)
        decode_tps = (token_chunks - 1) / decode_elapsed if token_chunks > 1 else 0.0

        # --- prefill rate ---------------------------------------------
        prefill_tps = None
        prompt_tokens = 0
        try:
            big = FILLER * 120
            t1 = time.time()
            resp = runtime.chat(
                handle,
                [{"role": "user", "content": big + "\n\nReply with the single word: ok"}],
                max_tokens=8,
                temperature=0.0,
            )
            prefill_s = time.time() - t1
            usage = resp.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            if prompt_tokens and prefill_s > 0:
                prefill_tps = round(prompt_tokens / prefill_s, 1)
        except Exception:
            pass

        watcher.stop()
        peak = max(watcher.peak, _rss_gb(handle.process))

        return PerfSample(
            ctx_size=ctx_size,
            load_seconds=round(load_s, 2),
            ttft_seconds=round(ttft, 3),
            decode_tps=round(decode_tps, 2),
            prefill_tps=prefill_tps,
            peak_rss_gb=round(peak, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=token_chunks,
        )

    except Exception as exc:
        return PerfSample(ctx_size, 0.0, 0.0, 0.0, None, 0.0, 0, 0, f"{type(exc).__name__}: {exc}")
    finally:
        watcher.stop()
        if handle is not None:
            try:
                runtime.stop(handle)
            except Exception:
                pass
            time.sleep(2)


def run(
    runtime: Runtime,
    model: Model,
    port: int = 8090,
    ctx_sizes: tuple[int, ...] = (4096, 8192, 16384),
) -> PerfReport:
    import platform

    report = PerfReport(
        model_id=model.id,
        name=model.name,
        arch=model.arch,
        params=model.total_params,
        active=model.active_params,
        runtime=model.runtime.kind,
        threads=model.runtime.threads or 0,
        host=f"{platform.system()} {platform.machine()}",
    )
    for ctx in ctx_sizes:
        print(f"llmlab: benchmarking {model.id} at ctx={ctx} ...")
        sample = measure_once(runtime, model, port, ctx)
        if sample.error:
            print(f"  ERROR: {sample.error}")
        else:
            print(
                f"  load={sample.load_seconds}s  ttft={sample.ttft_seconds}s  "
                f"decode={sample.decode_tps} tok/s  prefill={sample.prefill_tps} tok/s  "
                f"rss={sample.peak_rss_gb} GB"
            )
        report.samples.append(sample)
    return report
