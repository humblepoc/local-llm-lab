"""Model manifest: the plug-and-play contract.

A model is a directory containing ``model.toml``. Drop the directory into
``models/`` and the framework discovers it. No code changes, no registration
step, no imports to add.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Runtimes the framework knows how to drive.
KNOWN_RUNTIMES = ("llamacpp", "prism", "ollama")

# Evaluation tiers a model can participate in.
KNOWN_TIERS = ("perf", "coding", "tools")

VALID_ARCH = ("moe", "dense", "ternary")


class ManifestError(ValueError):
    """Raised when a model.toml is missing required fields or malformed."""


@dataclass(frozen=True)
class Source:
    """Where the weights come from. Weights are never committed to this repo."""

    repo: str
    file: str
    sha256: str | None = None
    size_gb: float | None = None

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"


@dataclass(frozen=True)
class RuntimeSpec:
    """How to serve this model.

    ``ctx_size`` is the *working* context, deliberately not the model's native
    context. A 262K-native model on a 32GB box will exhaust RAM; the manifest
    encodes something that actually fits.

    ``needs_build`` is a declaration, not a guess about the runtime kind. It is
    true only for a runtime that genuinely cannot be provisioned automatically.
    Ternary models once needed a from-source PrismML build, but that fork now
    ships prebuilt Windows x64 CPU binaries, so nothing in the default catalog
    sets this.
    """

    kind: str
    ctx_size: int = 8192
    threads: int = 0
    extra_args: tuple[str, ...] = ()
    needs_build: bool = False


@dataclass
class Model:
    id: str
    name: str
    org: str
    license: str
    arch: str
    total_params: str
    active_params: str | None
    context_native: int
    released: str | None
    description: str
    source: Source
    runtime: RuntimeSpec
    eval_tiers: tuple[str, ...]
    notes: str
    tags: tuple[str, ...]
    root: Path
    warnings: list[str] = field(default_factory=list)

    # -- paths -------------------------------------------------------------
    @property
    def weights_dir(self) -> Path:
        return self.root / "weights"

    @property
    def gguf_path(self) -> Path:
        return self.weights_dir / self.source.file

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    # -- state -------------------------------------------------------------
    @property
    def installed(self) -> bool:
        return self.gguf_path.exists()

    @property
    def verified(self) -> bool:
        """True when a checksum is declared and a .sha256 sidecar was written."""
        return (self.weights_dir / f"{self.source.file}.sha256").exists()

    @property
    def status(self) -> str:
        if self.runtime.needs_build:
            return "blocked" if not self.installed else "needs-build"
        return "ready" if self.installed else "not-installed"

    @property
    def approx_ram_gb(self) -> float | None:
        """Weights plus a rough KV cache allowance at the configured context."""
        if not self.source.size_gb:
            return None
        # KV cache scales with context. Rough allowance: 1GB per 32k tokens
        # for a 30B-class model at 4-bit, scaled down for smaller models.
        ctx_gb = (self.runtime.ctx_size / 32768) * 1.0
        return round(self.source.size_gb + ctx_gb, 1)

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> Model:
        """Parse and validate a model.toml. Raises ManifestError on problems."""
        try:
            # Tolerate a UTF-8 BOM: PowerShell 5.1's `Set-Content -Encoding UTF8`
            # and Windows Notepad both write one, and tomllib rejects it.
            raw_text = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError as exc:
            raise ManifestError(f"manifest not found: {path}") from exc
        try:
            raw = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as exc:
            raise ManifestError(f"{path}: invalid TOML: {exc}") from exc

        warnings: list[str] = []

        def section(name: str) -> dict:
            value = raw.get(name)
            if value is None:
                raise ManifestError(f"{path}: missing required [{name}] section")
            return value

        def need(data: dict, key: str, where: str) -> object:
            if key not in data:
                raise ManifestError(f"{path}: missing required key '{key}' in [{where}]")
            return data[key]

        m = section("model")
        s = section("source")
        r = section("runtime")

        src = Source(
            repo=str(need(s, "repo", "source")),
            file=str(need(s, "file", "source")),
            sha256=s.get("sha256"),
            size_gb=float(s["size_gb"]) if s.get("size_gb") else None,
        )

        kind = str(need(r, "kind", "runtime"))
        if kind not in KNOWN_RUNTIMES:
            warnings.append(f"unknown runtime '{kind}' (known: {', '.join(KNOWN_RUNTIMES)})")

        arch = str(m.get("arch", "dense"))
        if arch not in VALID_ARCH:
            warnings.append(f"unknown arch '{arch}' (known: {', '.join(VALID_ARCH)})")

        tiers = tuple(m.get("eval_tiers", ("perf",)))
        bad = [t for t in tiers if t not in KNOWN_TIERS]
        if bad:
            warnings.append(f"unknown eval tier(s): {', '.join(bad)}")

        model_id = str(need(m, "id", "model"))
        if model_id != path.parent.name:
            warnings.append(
                f"model.id '{model_id}' does not match directory name '{path.parent.name}'"
            )

        runtime = RuntimeSpec(
            kind=kind,
            ctx_size=int(r.get("ctx_size", 8192)),
            threads=int(r.get("threads", 0)),
            extra_args=tuple(r.get("extra_args", ())),
            needs_build=bool(r.get("needs_build", False)),
        )

        return cls(
            id=model_id,
            name=str(m.get("name", model_id)),
            org=str(m.get("org", "unknown")),
            license=str(m.get("license", "unknown")),
            arch=arch,
            total_params=str(m.get("total_params", "?")),
            active_params=m.get("active_params"),
            context_native=int(m.get("context_native", 0)),
            released=m.get("released"),
            description=str(m.get("description", "")),
            source=src,
            runtime=runtime,
            eval_tiers=tiers,
            notes=str(m.get("notes", "")),
            tags=tuple(m.get("tags", ())),
            root=path.parent,
            warnings=warnings,
        )
