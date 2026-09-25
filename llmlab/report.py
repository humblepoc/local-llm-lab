"""Leaderboard generation.

Reads results/raw/*.json and renders results/leaderboard.md. The raw JSON is
gitignored; the rendered leaderboard is committed so the repo carries its own
results without shipping megabytes of samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import Model
from .registry import RAW_RESULTS_DIR, RESULTS_DIR

LEADERBOARD = RESULTS_DIR / "leaderboard.md"


@dataclass
class Row:
    model_id: str
    name: str
    params: str
    arch: str
    license: str = "?"
    decode_tps: float | None = None
    ttft: float | None = None
    prefill_tps: float | None = None
    peak_rss_gb: float | None = None
    load_s: float | None = None
    coding_pct: float | None = None
    tools_pct: float | None = None
    failure: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_quality(self) -> bool:
        return self.coding_pct is not None or self.tools_pct is not None

    @property
    def has_measurement(self) -> bool:
        return self.decode_tps is not None or self.failure is not None

    def quality_score(self) -> float | None:
        """Simple mean of available quality signals, 0-100."""
        vals = [v for v in (self.coding_pct, self.tools_pct) if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    def fits_box(self) -> str:
        """Verdict against a 32GB CPU-only machine."""
        if self.failure:
            return "❌ failed"
        if self.peak_rss_gb is None:
            return "not measured"
        if self.peak_rss_gb > 30:
            return "❌ too tight"
        if self.peak_rss_gb > 26:
            return "⚠️ tight"
        return "✅ fits"


def _fmt(v: float | None, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


def load_raw() -> tuple[dict[str, dict], list[str]]:
    """Return ({model_id: {kind: payload}}, list of parse warnings)."""
    data: dict[str, dict] = {}
    warns: list[str] = []
    if not RAW_RESULTS_DIR.is_dir():
        return data, warns
    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warns.append(f"{path.name}: {exc}")
            continue
        stem = path.stem
        if "--" not in stem:
            warns.append(f"{path.name}: unexpected filename (expected <model>--<kind>.json)")
            continue
        model_id, kind = stem.split("--", 1)
        data.setdefault(model_id, {})[kind] = payload
    return data, warns


def build_rows(models: list[Model]) -> list[Row]:
    raw, _ = load_raw()
    rows: list[Row] = []
    for model in models:
        entry = raw.get(model.id, {})
        row = Row(
            model_id=model.id,
            name=model.name,
            params=model.total_params,
            arch=model.arch,
            license=model.license,
        )
        perf = entry.get("perf")
        if perf and perf.get("samples"):
            ok = [s for s in perf["samples"] if not s.get("error")]
            if ok:
                row.decode_tps = round(max(s["decode_tps"] for s in ok), 2)
                row.ttft = round(min(s["ttft_seconds"] for s in ok), 3)
                row.peak_rss_gb = round(max(s["peak_rss_gb"] for s in ok), 2)
                row.load_s = round(sum(s["load_seconds"] for s in ok) / len(ok), 2)
                pf = [s["prefill_tps"] for s in ok if s.get("prefill_tps")]
                if pf:
                    row.prefill_tps = round(max(pf), 1)
            else:
                # Every sample failed. That is still a result and belongs on
                # the board, not silently omitted.
                first_err = next(
                    (s.get("error") for s in perf["samples"] if s.get("error")), None
                )
                row.failure = (first_err or "run failed")[:160]
                rss = [s["peak_rss_gb"] for s in perf["samples"] if s.get("peak_rss_gb")]
                if rss:
                    row.peak_rss_gb = round(max(rss), 2)
                ld = [s["load_seconds"] for s in perf["samples"] if s.get("load_seconds")]
                if ld:
                    row.load_s = round(max(ld), 2)
                pfe = [s["prefill_tps"] for s in perf["samples"] if s.get("prefill_tps")]
                if pfe:
                    row.prefill_tps = round(max(pfe), 1)
        coding = entry.get("coding")
        if coding and coding.get("total"):
            row.coding_pct = round(100 * coding.get("passed", 0) / coding["total"], 1)
        tools = entry.get("tools")
        if tools and tools.get("total"):
            row.tools_pct = round(100 * tools.get("passed", 0) / tools["total"], 1)
        rows.append(row)
    return rows


def render(rows: list[Row], models: list[Model]) -> str:
    measured = [r for r in rows if r.has_measurement or r.has_quality]
    lines: list[str] = []
    lines.append("# Leaderboard")
    lines.append("")
    lines.append(
        "Auto-generated by `llmlab rank`. Raw samples live in `results/raw/` "
        "(gitignored); this file is committed."
    )
    lines.append("")

    if not measured:
        lines.append("_No results yet. Run `llmlab run <model>` to populate._")
        lines.append("")
    else:
        lines.append("## Results")
        lines.append("")
        lines.append(
            "| Model | Params | Arch | Decode tok/s | TTFT | Prefill tok/s | Peak RSS | Coding | Tools | Fits 32GB |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        # Ranked by quality then speed, with failures pinned to the bottom
        # rather than dropped.
        ordered = sorted(
            measured,
            key=lambda r: (
                r.failure is not None,
                -(r.quality_score() or 0),
                -(r.decode_tps or 0),
            ),
        )
        for r in ordered:
            lines.append(
                f"| `{r.model_id}` | {r.params} | {r.arch} | {_fmt(r.decode_tps)} | "
                f"{_fmt(r.ttft, 's')} | {_fmt(r.prefill_tps)} | "
                f"{_fmt(r.peak_rss_gb, ' GB')} | {_fmt(r.coding_pct, '%')} | "
                f"{_fmt(r.tools_pct, '%')} | {r.fits_box()} |"
            )
        lines.append("")

        failures = [r for r in ordered if r.failure]
        if failures:
            lines.append("### Failed runs")
            lines.append("")
            for r in failures:
                lines.append(f"- `{r.model_id}`: {r.failure}")
            lines.append("")

    lines.append("## Catalog")
    lines.append("")
    lines.append("| Model | Params | Arch | License | Runtime | Status |")
    lines.append("|---|---|---|---|---|---|")
    for m in sorted(models, key=lambda m: m.id):
        active = f" ({m.active_params} active)" if m.active_params else ""
        lines.append(
            f"| `{m.id}` | {m.total_params}{active} | {m.arch} | {m.license} | "
            f"{m.runtime.kind} | {m.status} |"
        )
    lines.append("")

    warns = [w for m in models for w in m.warnings]
    if warns:
        lines.append("## Manifest warnings")
        lines.append("")
        for w in warns:
            lines.append(f"- `{m.id}`: {w}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Speed figures are single-run measurements on the machine that produced them. "
        "They are not comparable across hardware. See each model's README for setup notes."
    )
    lines.append("")
    return "\n".join(lines)


def generate(models: list[Model]) -> Path:
    rows = build_rows(models)
    _, warns = load_raw()
    text = render(rows, models)
    if warns:
        text += "\n<!-- raw parse warnings: " + "; ".join(warns) + " -->\n"
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    LEADERBOARD.write_text(text, encoding="utf-8")
    return LEADERBOARD
