"""Model discovery.

Scans ``models/*/model.toml``. That is the entire registration mechanism —
adding a model never requires editing Python.
"""

from __future__ import annotations

from pathlib import Path

from .manifest import ManifestError, Model

# Repo root = parent of the llmlab package.
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
TASKS_DIR = REPO_ROOT / "tasks"
RESULTS_DIR = REPO_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
CACHE_DIR = REPO_ROOT / ".llmlab"
BIN_DIR = CACHE_DIR / "bin"


class Registry:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or MODELS_DIR
        self._models: dict[str, Model] = {}
        self.errors: list[str] = []
        self.load()

    def load(self) -> None:
        self._models.clear()
        self.errors.clear()
        if not self.models_dir.is_dir():
            return
        for manifest in sorted(self.models_dir.glob("*/model.toml")):
            try:
                model = Model.load(manifest)
            except ManifestError as exc:
                self.errors.append(str(exc))
                continue
            self._models[model.id] = model

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models.values())

    def all(self) -> list[Model]:
        return sorted(self._models.values(), key=lambda m: m.id)

    def get(self, model_id: str) -> Model:
        try:
            return self._models[model_id]
        except KeyError:
            known = ", ".join(sorted(self._models)) or "(none)"
            raise KeyError(f"unknown model '{model_id}'. known: {known}") from None

    def find(self, fragment: str) -> list[Model]:
        """Loose match so `llmlab bench laguna` works."""
        frag = fragment.lower()
        return [m for m in self.all() if frag in m.id.lower() or frag in m.name.lower()]

    def resolve(self, fragment: str) -> Model:
        """Resolve an id or unique fragment; raise if ambiguous or missing."""
        if fragment in self._models:
            return self._models[fragment]
        hits = self.find(fragment)
        if not hits:
            known = ", ".join(sorted(self._models)) or "(none)"
            raise KeyError(f"no model matches '{fragment}'. known: {known}")
        if len(hits) > 1:
            names = ", ".join(m.id for m in hits)
            raise KeyError(f"'{fragment}' is ambiguous: {names}")
        return hits[0]
