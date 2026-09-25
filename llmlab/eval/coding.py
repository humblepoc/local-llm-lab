"""Coding capability evaluation.

Each task is a prompt plus executable assertions. The model's code is exec'd in
a *subprocess* with a timeout, so a runaway loop or a stray ``exit()`` cannot
take the harness down.

Honest scope note: this is a small auto-graded suite. It reliably separates
"can write a function" from "cannot", and it will systematically underrate
agentic ability (multi-file editing, tool loops, debugging) because it never
exercises those. Treat it as one signal, not a ranking oracle.

Task format (tasks/coding/*.json)::

    {
      "id": "sum_list",
      "tags": ["python", "basic"],
      "prompt": "Write a function sum_list(xs) ...",
      "check": "assert sum_list([1, 2, 3]) == 6"
    }
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..manifest import Model
from ..registry import TASKS_DIR
from ..runtimes.base import Runtime, ServerHandle

CODING_DIR = TASKS_DIR / "coding"
DEFAULT_TIMEOUT = 20

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    seconds: float
    error: str | None = None
    output_len: int = 0


@dataclass
class EvalReport:
    model_id: str
    name: str
    kind: str
    total: int = 0
    passed: int = 0
    invalid: int = 0
    mean_seconds: float | None = None
    results: list[TaskResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return round(100 * self.passed / self.total, 1) if self.total else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pass_rate"] = self.pass_rate
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_tasks(directory: Path = CODING_DIR) -> list[dict]:
    if not directory.is_dir():
        return []
    tasks: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                tasks.extend(data)
            else:
                tasks.append(data)
        except json.JSONDecodeError as exc:
            print(f"llmlab: skipping malformed task file {path.name}: {exc}")
    return [t for t in tasks if t.get("prompt") and t.get("check")]


def extract_code(text: str) -> str:
    """Pull code out of a markdown-fenced or bare response."""
    if not text:
        return ""
    blocks = _FENCE.findall(text)
    if blocks:
        # Prefer the longest block — models sometimes emit a short example first.
        return max(blocks, key=len).strip()
    # No fences: assume the whole thing is code.
    return text.strip()


def _run_check(code: str, check: str, timeout: int) -> tuple[bool, str | None]:
    """Execute ``code`` then ``check`` in an isolated subprocess."""
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "candidate.py"
        script.write_text(code + "\n\n# --- checks ---\n" + check + "\n", encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        if proc.returncode == 0:
            return True, None
        err = (proc.stderr or "").strip().splitlines()
        return False, (err[-1][:200] if err else f"exit {proc.returncode}")


def run_task(
    runtime: Runtime,
    handle: ServerHandle,
    task: dict,
    *,
    max_tokens: int = 900,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
) -> TaskResult:
    t0 = time.time()
    try:
        resp = runtime.chat(
            handle,
            [{"role": "user", "content": task["prompt"]}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choices = resp.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
        code = extract_code(text)
        if not code:
            return TaskResult(task["id"], False, time.time() - t0, "empty response", 0)
        ok, err = _run_check(code, task["check"], timeout)
        return TaskResult(task["id"], ok, round(time.time() - t0, 2), err, len(code))
    except Exception as exc:
        return TaskResult(task["id"], False, time.time() - t0, f"{type(exc).__name__}: {exc}")


def run(
    runtime: Runtime,
    model: Model,
    port: int = 8090,
    limit: int | None = None,
) -> EvalReport:
    tasks = load_tasks()
    if not tasks:
        raise RuntimeError(f"no coding tasks found in {CODING_DIR}")
    if limit:
        tasks = tasks[:limit]

    report = EvalReport(model_id=model.id, name=model.name, kind="coding", total=len(tasks))

    handle: ServerHandle | None = None
    try:
        handle = runtime.start(model, port)
        print(f"llmlab: waiting for {model.id}...")
        if not runtime.wait_ready(handle, timeout=900):
            err = "server did not become healthy"
            if runtime.__class__.__name__ == "LlamaCppRuntime":
                err += "\n" + runtime.tail_log(handle)
            raise RuntimeError(err)

        print(f"llmlab: running {len(tasks)} coding tasks against {model.id}\n")
        for i, task in enumerate(tasks, 1):
            res = run_task(runtime, handle, task)
            report.results.append(res)
            if res.passed:
                report.passed += 1
            mark = "PASS" if res.passed else "FAIL"
            print(f"  [{i:>2}/{len(tasks)}] {mark}  {res.task_id}  ({res.seconds}s)")
            if not res.passed and res.error:
                print(f"          {res.error}")

    finally:
        if handle is not None:
            try:
                runtime.stop(handle)
            except Exception:
                pass
            time.sleep(2)

    times = [r.seconds for r in report.results]
    report.mean_seconds = round(sum(times) / len(times), 2) if times else None
    print(f"\nllmlab: {report.passed}/{report.total} passed ({report.pass_rate}%)")
    return report
