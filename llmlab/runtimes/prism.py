"""PrismML llama.cpp fork — required for ternary models (Ternary Bonsai).

Stock llama.cpp refuses the PQ2_0 / PTQ1_0 packs these models ship in: the
ternary kernels and the required Hadamard rotation are not upstream. As of
September 2026 they are still not in mainline (ggml-org/llama.cpp discussion
#22019), so this runtime cannot self-install and will not pretend otherwise.

The fork has no official Windows release binaries, so serving one of these
means building from source with CMake and a C++ toolchain. The framework
reports that as `blocked` rather than failing partway through an eval.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..manifest import Model
from ..registry import CACHE_DIR
from .base import Runtime, RuntimeError_, ServerHandle

FORK_URL = "https://github.com/PrismML-Eng/llama.cpp"
PRISM_BIN_DIR = CACHE_DIR / "prism"


class PrismRuntime(Runtime):
    """Serves ternary models via a user-built PrismML llama.cpp."""

    kind = "prism"

    def __init__(self) -> None:
        self.server_exe: Path | None = None

    def needs_build(self) -> bool:
        return True

    def install_hint(self) -> str:
        return (
            "Ternary models need a from-source build of the PrismML llama.cpp fork.\n"
            f"    1. Install CMake and a C++ toolchain (MSVC Build Tools)\n"
            f"    2. git clone {FORK_URL}\n"
            "    3. cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release -j\n"
            f"    4. Copy build/bin/llama-server.exe into {PRISM_BIN_DIR}\n"
            "  Stock llama.cpp and Ollama cannot load PQ2_0/PTQ1_0 packs."
        )

    def _find_server(self) -> Path | None:
        target = PRISM_BIN_DIR / "llama-server.exe"
        if target.exists():
            return target
        if PRISM_BIN_DIR.is_dir():
            for candidate in PRISM_BIN_DIR.rglob("llama-server.exe"):
                return candidate
        return None

    def ensure_installed(self) -> None:
        found = self._find_server()
        if found:
            self.server_exe = found
            return
        raise RuntimeError_(
            "PrismML fork not built.\n" + self.install_hint()
        )

    def start(
        self, model: Model, port: int, ctx_size: int | None = None
    ) -> ServerHandle:
        if not self.server_exe or not self.server_exe.exists():
            self.ensure_installed()
        assert self.server_exe is not None

        if not model.installed:
            raise RuntimeError_(
                f"weights not found for '{model.id}': {model.gguf_path}\n"
                f"run:  llmlab setup {model.id}"
            )

        # The Prism fork also serves on the OpenAI-compatible endpoint, so we
        # deliberately do not subclass LlamaCppRuntime — the pack format and
        # the Hadamard rotation are fork-specific and stock flags differ.
        raise RuntimeError_(
            "PrismML runtime launching is not wired up yet. "
            "Build the fork first; see `llmlab doctor`."
        )
