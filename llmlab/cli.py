"""llmlab — plug-and-play local model evaluation.

Commands:
    llmlab doctor              environment audit
    llmlab list                show every known model
    llmlab new-model <id>      scaffold a model directory
    llmlab setup <id>          download + verify weights
    llmlab serve <id>          start a server
    llmlab bench <id>          performance suite
    llmlab eval <id>           coding + tools quality suite
    llmlab run <id>            bench + eval
    llmlab rank                rebuild results/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import report as report_mod
from . import weights
from .bench import perf
from .eval import coding, tools
from .manifest import Model
from .registry import MODELS_DIR, RAW_RESULTS_DIR, REPO_ROOT, Registry
from .runtimes import get_runtime
from .runtimes.base import RuntimeError_

MODELS = Registry()


# ---------------------------------------------------------------- helpers
def _resolve(fragment: str) -> Model:
    try:
        return MODELS.resolve(fragment)
    except KeyError as exc:
        print(f"llmlab: {exc}")
        raise SystemExit(2)


def _runtime_for(model: Model):
    return get_runtime(model.runtime.kind)


def _save(payload: dict, model_id: str, kind: str) -> Path:
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_RESULTS_DIR / f"{model_id}--{kind}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- doctor
def _cpu_flags() -> set[str]:
    """Detect AVX-VNNI, which materially speeds up quantised matmul in llama.cpp."""
    flags: set[str] = set()
    try:
        import ctypes

        class CPUID(ctypes.Structure):
            _fields_ = [
                ("eax", ctypes.c_uint32), ("ebx", ctypes.c_uint32),
                ("ecx", ctypes.c_uint32), ("edx", ctypes.c_uint32),
            ]

        for leaf, sub in ((7, 0),):
            cpuid = CPUID()
            ctypes.windll.kernel32.__cpuid  # type: ignore[attr-defined]
            # Portable-enough check via /proc-free route: fall back to known set.
        # Windows has no /proc/cpuinfo; use the simplest reliable signal instead.
    except Exception:
        pass
    # PowerShell is the reliable route on Windows.
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "$env:PROCESSOR_IDENTIFIER"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        if out:
            flags.add(out.strip())
    except Exception:
        pass
    return flags


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("llmlab doctor")
    print("=" * 60)

    print(f"\nPython      {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform    {platform.platform()}")
    print(f"Repo        {REPO_ROOT}")
    print(f"CPU         {platform.processor() or platform.machine()}")
    print(f"Cores       {os.cpu_count()}")

    # RAM
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        print(f"RAM         {st.ullTotalPhys/1_073_741_824:.1f} GB total, "
              f"{st.ullAvailPhys/1_073_741_824:.1f} GB available")
    except Exception:
        print("RAM         could not determine")

    # Disk
    try:
        usage = shutil.disk_usage(REPO_ROOT)
        print(f"Disk        {usage.free/1_073_741_824:.0f} GB free at {REGISTRY}")
    except Exception:
        pass

    print("\nExternal tools")
    for tool, why in (
        ("git", "version control / pushing results"),
        ("cmake", "only needed to build the PrismML fork for ternary models"),
        ("winget", "package installs"),
    ):
        path = shutil.which(tool)
        mark = "OK  " if path else "MISS"
        print(f"  [{mark}] {tool:<8} {why}")
        if path:
            print(f"           {path}")

    print("\nOptional Python packages")
    for mod, why in (
        ("psutil", "peak-RSS measurement in bench (results degrade to 0 without it)"),
        ("huggingface_hub", "alternative weight downloader"),
    ):
        try:
            __import__(mod)
            print(f"  [OK  ] {mod:<18} {why}")
        except ImportError:
            print(f"  [--  ] {mod:<18} {why}  (pip install {mod})")

    print("\nllama.cpp runtime")
    try:
        rt = get_runtime("llamacpp")
        found = rt._find_server()
        if found:
            print(f"  [OK  ] llama-server at {found}")
            ver = rt.version
            if ver:
                print(f"         {ver}")
        else:
            print("  [--  ] not downloaded yet (llmlab setup <model> fetches it)")
    except Exception as exc:
        print(f"  [ERR ] {exc}")

    print("\nModels")
    if not len(MODELS):
        print("  (none)")
    for m in MODELS:
        warn = f"  ! {len(m.warnings)} warning(s)" if m.warnings else ""
        print(f"  {m.status:<14} {m.id:<30} {m.total_params}{warn}")
    for err in MODELS.errors:
        print(f"  [ERR ] {err}")

    # Per-model blockers
    blocked = [m for m in MODELS if m.runtime.needs_build]
    if blocked:
        print("\nBlocked runtimes")
        for m in blocked:
            rt = get_runtime(m.runtime.kind)
            print(f"  {m.id} needs '{m.runtime.kind}':")
            for line in rt.install_hint().splitlines():
                print(f"    {line}")

    print()
    return 0


# ---------------------------------------------------------------- list
def cmd_list(_args: argparse.Namespace) -> int:
    if not len(MODELS):
        print("No models found. Scaffold one with: llmlab new-model <id>")
    else:
        print(f"{'STATUS':<15}{'MODEL':<30}{'PARAMS':<16}{'ARCH':<9}{'CTX':<8}LICENSE")
        print("-" * 100)
        for m in MODELS.all():
            active = f"/{m.active_params}" if m.active_params else ""
            print(
                f"{m.status:<15}{m.id:<30}{m.total_params + active:<16}"
                f"{m.arch:<9}{m.runtime.ctx_size:<8}{m.license}"
            )
            if m.notes:
                print(f"{'':<15}  {m.notes[:78]}")
    # Always surface manifest problems, even when nothing loaded.
    for err in MODELS.errors:
        print(f"  manifest error: {err}")
    return 0


# ---------------------------------------------------------------- new-model
TEMPLATE = '''[model]
id = "{id}"
name = "{name}"
org = "TODO"
license = "TODO"
arch = "dense"            # moe | dense | ternary
total_params = "TODO"
active_params = ""        # MoE only, e.g. "3B"
context_native = 0
released = ""
description = "TODO"
eval_tiers = ["perf", "coding", "tools"]
notes = ""
tags = []

[source]
repo = "TODO/TODO-GGUF"
file = "TODO.gguf"
sha256 = ""               # strongly recommended
size_gb = 0.0

[runtime]
kind = "llamacpp"         # llamacpp | prism
ctx_size = 8192
threads = 0               # 0 = all cores
extra_args = []
'''

README_TEMPLATE = """# {name}

> Status: **TODO** — fill in `model.toml`, then run `llmlab setup {id}`.

## What it is

TODO

## Why it is on this machine

TODO — tie the choice to the 32GB / CPU-only constraint.

## Setup

```powershell
llmlab setup {id}
```

Weights are **not** in this repository. They are fetched from
`{repo}` on demand.

## Measured results

| Metric | Value |
|---|---|
| Decode tok/s | — |
| TTFT | — |
| Peak RSS | — |
| Coding pass@1 | — |
| Tool-call validity | — |

Run `llmlab run {id}` to populate, then `llmlab rank`.

## Notes and gotchas

TODO
"""


def cmd_new_model(args: argparse.Namespace) -> int:
    model_id = args.id
    root = MODELS_DIR / model_id
    if (root / "model.toml").exists():
        print(f"llmlab: {root/'model.toml'} already exists")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    (root / "weights").mkdir(exist_ok=True)
    (root / "model.toml").write_text(
        TEMPLATE.format(id=model_id, name=args.name or model_id), encoding="utf-8"
    )
    (root / "README.md").write_text(
        README_TEMPLATE.format(
            name=args.name or model_id,
            id=model_id,
            repo="TODO/TODO-GGUF",
        ),
        encoding="utf-8",
    )
    MODELS.load()
    print(f"llmlab: scaffolded {root}")
    print(f"        1. edit model.toml  2. llmlab setup {model_id}  3. llmlab run {model_id}")
    return 0


# ---------------------------------------------------------------- setup
def cmd_setup(args: argparse.Namespace) -> int:
    model = _resolve(args.id)
    rt = _runtime_for(model)
    if rt.needs_build():
        print(f"llmlab: '{model.id}' needs the '{model.runtime.kind}' runtime, which must be built from source.")
        print(rt.install_hint())
        print("\nllmlab: continuing with weight download only.")
    try:
        rt.ensure_installed()
    except Exception as exc:
        print(f"llmlab: runtime not ready: {exc}")
    try:
        weights.fetch(model, force=args.force)
    except Exception as exc:
        print(f"llmlab: weight fetch failed: {exc}")
        return 1
    print(f"llmlab: {model.id} is {model.status}")
    return 0


# ---------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    model = _resolve(args.id)
    rt = _runtime_for(model)
    rt.ensure_installed()
    handle = rt.start(model, args.port)
    print(f"llmlab: {model.id} starting on {handle.base_url} (Ctrl-C to stop)")
    try:
        while not rt.health(handle):
            time.sleep(2)
        print(f"llmlab: ready — {handle.base_url}/v1")
        while True:
            time.sleep(5)
            if handle.process and handle.process.poll() is not None:
                print("\nllmlab: server exited")
                break
    except KeyboardInterrupt:
        print("\nllmlab: stopping")
    finally:
        rt.stop(handle)
    return 0


# ---------------------------------------------------------------- bench
def cmd_bench(args: argparse.Namespace) -> int:
    model = _resolve(args.id)
    rt = _runtime_for(model)
    rt.ensure_installed()
    ctxs = tuple(int(c) for c in args.ctx.split(",")) if args.ctx else (4096, 8192, 16384)
    rep = perf.run(rt, model, port=args.port, ctx_sizes=ctxs)
    path = _save(rep.to_dict(), model.id, "perf")
    print(f"\nllmlab: wrote {path}")
    print(f"  best decode   {rep.best_decode} tok/s")
    print(f"  mean decode   {rep.mean_decode} tok/s")
    print(f"  mean load     {rep.mean_load} s")
    print(f"  peak RSS      {rep.max_peak_rss} GB")
    return 0


# ---------------------------------------------------------------- eval
def cmd_eval(args: argparse.Namespace) -> int:
    model = _resolve(args.id)
    rt = _runtime_for(model)
    rt.ensure_installed()
    ok = True
    if "coding" in model.eval_tiers:
        try:
            rep = coding.run(rt, model, port=args.port, limit=args.limit)
            _save(rep.to_dict(), model.id, "coding")
            print(f"llmlab: coding pass@1 {rep.pass_rate}%")
        except Exception as exc:
            print(f"llmlab: coding eval failed: {exc}")
            ok = False
    if "tools" in model.eval_tiers:
        try:
            trep = tools.run(rt, model, port=args.port)
            _save(trep.to_dict(), model.id, "tools")
            print(f"llmlab: tool-call validity {trep.pass_rate}%")
        except Exception as exc:
            print(f"llmlab: tool eval failed: {exc}")
            ok = False
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    rc = cmd_bench(args)
    args.limit = None
    rc2 = cmd_eval(args)
    cmd_rank(argparse.Namespace())
    return rc or rc2


# ---------------------------------------------------------------- rank
def cmd_rank(_args: argparse.Namespace) -> int:
    path = report_mod.generate(MODELS.all())
    print(f"llmlab: wrote {path}")
    for err in MODELS.errors:
        print(f"  manifest error: {err}")
    return 0


# ---------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="environment audit").set_defaults(func=cmd_doctor)
    sub.add_parser("list", help="show all models").set_defaults(func=cmd_list)
    sub.add_parser("rank", help="rebuild leaderboard").set_defaults(func=cmd_rank)

    nm = sub.add_parser("new-model", help="scaffold a model directory")
    nm.add_argument("id")
    nm.add_argument("--name", default=None)
    nm.set_defaults(func=cmd_new_model)

    for name, fn, helptext in (
        ("setup", cmd_setup, "download + verify weights"),
        ("serve", cmd_serve, "start a server"),
        ("bench", cmd_bench, "performance suite"),
        ("eval", cmd_eval, "quality suite"),
        ("run", cmd_run, "bench + eval"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("id", help="model id or unique fragment")
        sp.add_argument("--port", type=int, default=8090)
        sp.add_argument("--force", action="store_true", help="re-download weights")
        if name in ("bench",):
            sp.add_argument("--ctx", default=None, help="comma-separated ctx sizes")
        if name in ("eval", "run"):
            sp.add_argument("--limit", type=int, default=None, help="cap coding tasks")
        sp.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError_ as exc:
        print(f"llmlab: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
