"""Tool-calling validity evaluation.

This is the closest proxy we have for "can OpenCode actually drive this model".
Every OpenCode session is a tool loop, so a model that cannot emit a
schema-valid ``tool_calls`` block is unusable here regardless of how well it
scores on function-writing tasks.

Tasks (tasks/tools/*.json)::

    {
      "id": "read_file",
      "prompt": "Read the file config.json and tell me what it contains.",
      "tools": [ ... OpenAI tool schema ... ],
      "expect": { "name": "read_file" }          # optional
      "require_args": ["path"]                    # optional
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..manifest import Model
from ..registry import TASKS_DIR
from ..runtimes.base import Runtime, ServerHandle

TOOLS_DIR = TASKS_DIR / "tools"


@dataclass
class ToolResult:
    task_id: str
    passed: bool
    called_tool: str | None
    args_valid: bool
    seconds: float
    error: str | None = None


@dataclass
class ToolReport:
    model_id: str
    name: str
    kind: str
    total: int = 0
    passed: int = 0
    no_tool_call: int = 0
    bad_args: int = 0
    mean_seconds: float | None = None
    results: list[ToolResult] = field(default_factory=list)
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


def load_tasks(directory: Path = TOOLS_DIR) -> list[dict]:
    if not directory.is_dir():
        return []
    tasks: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tasks.extend(data if isinstance(data, list) else [data])
        except json.JSONDecodeError as exc:
            print(f"llmlab: skipping malformed task file {path.name}: {exc}")
    return [t for t in tasks if t.get("prompt") and t.get("tools")]


def _args_are_valid(tool: dict, args: dict, require: list[str]) -> bool:
    """Shallow schema check: required keys present, declared types respected."""
    for key in require:
        if key not in args:
            return False
    schema = (tool.get("function") or {}).get("parameters") or {}
    props = schema.get("properties") or {}
    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        want = spec.get("type")
        if want == "string" and not isinstance(value, str):
            return False
        if want == "number" and not isinstance(value, (int, float)):
            return False
        if want == "integer" and not isinstance(value, int):
            return False
        if want == "boolean" and not isinstance(value, bool):
            return False
        if want == "array" and not isinstance(value, list):
            return False
        if want == "object" and not isinstance(value, dict):
            return False
    return True


def run_task(
    runtime: Runtime, handle: ServerHandle, task: dict, *, max_tokens: int = 400
) -> ToolResult:
    t0 = time.time()
    try:
        resp = runtime.chat(
            handle,
            [{"role": "user", "content": task["prompt"]}],
            max_tokens=max_tokens,
            temperature=0.0,
            tools=task["tools"],
        )
        choices = resp.get("choices") or []
        msg = (choices[0].get("message") or {}) if choices else {}
        calls = msg.get("tool_calls") or []

        if not calls:
            return ToolResult(
                task["id"], False, None, False, round(time.time() - t0, 2),
                "no tool_calls emitted",
            )

        call = calls[0]
        fn = call.get("function") or {}
        name = fn.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            return ToolResult(
                task["id"], False, name, False, round(time.time() - t0, 2),
                "arguments not valid JSON",
            )
        if not isinstance(args, dict):
            return ToolResult(
                task["id"], False, name, False, round(time.time() - t0, 2),
                "arguments not an object",
            )

        expect = task.get("expect") or {}
        want_name = expect.get("name")
        if want_name and name != want_name:
            return ToolResult(
                task["id"], False, name, False, round(time.time() - t0, 2),
                f"called '{name}', expected '{want_name}'",
            )

        ok = _args_are_valid(task["tools"][0], args, task.get("require_args") or [])
        return ToolResult(
            task["id"], ok, name, ok, round(time.time() - t0, 2),
            None if ok else "arguments failed schema check",
        )
    except Exception as exc:
        return ToolResult(
            task["id"], False, None, False, time.time() - t0, f"{type(exc).__name__}: {exc}"
        )


def run(runtime: Runtime, model: Model, port: int = 8090) -> ToolReport:
    tasks = load_tasks()
    if not tasks:
        raise RuntimeError(f"no tool tasks found in {TOOLS_DIR}")

    report = ToolReport(model_id=model.id, name=model.name, kind="tools", total=len(tasks))
    handle: ServerHandle | None = None
    try:
        handle = runtime.start(model, port)
        print(f"llmlab: waiting for {model.id}...")
        if not runtime.wait_ready(handle, timeout=900):
            err = "server did not become healthy"
            if runtime.__class__.__name__ == "LlamaCppRuntime":
                err += "\n" + runtime.tail_log(handle)
            raise RuntimeError(err)

        print(f"llmlab: running {len(tasks)} tool-call tasks against {model.id}\n")
        for i, task in enumerate(tasks, 1):
            res = run_task(runtime, handle, task)
            report.results.append(res)
            if res.passed:
                report.passed += 1
            else:
                if res.called_tool is None:
                    report.no_tool_call += 1
                elif not res.args_valid:
                    report.bad_args += 1
            mark = "PASS" if res.passed else "FAIL"
            print(f"  [{i:>2}/{len(tasks)}] {mark}  {res.task_id} -> {res.called_tool} ({res.seconds}s)")
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
    print(f"\nllmlab: {report.passed}/{report.total} valid tool calls ({report.pass_rate}%)")
    return report
