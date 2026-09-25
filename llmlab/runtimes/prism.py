"""PrismML llama.cpp fork — required for ternary models (Ternary Bonsai).

Stock llama.cpp and Ollama cannot load PrismML's `PQ2_0` / `PTQ1_0` packs: the
ternary kernels and the required Hadamard rotation are not upstream. They were
still missing from mainline as of September 2026 (ggml-org/llama.cpp discussion
#22019).

**Correction to an earlier claim in this repo:** the fork was previously
documented as having no prebuilt Windows binaries, requiring a from-source
CMake + MSVC build. That is wrong. PrismML publishes a full CI matrix, and the
latest release ships `llama-prism-<tag>-bin-win-cpu-x64.zip`. No compiler
toolchain is needed, and this runtime fetches the binary exactly like the stock
`llamacpp` runtime does.

The served API is still OpenAI-compatible; only the weight pack format differs.
So this subclasses `LlamaCppRuntime` and inherits the whole lifecycle.
"""

from __future__ import annotations

import re

from ..registry import CACHE_DIR
from .base import register_runtime
from .llamacpp import LlamaCppRuntime

FORK_URL = "https://github.com/PrismML-Eng/llama.cpp"
PRISM_BIN_ROOT = CACHE_DIR / "prism"

# Prism release tags look like `prism-b10735-842b188`; pull the build number out.
_TAG_RE = re.compile(r"prism-b(\d+)")


@register_runtime
class PrismRuntime(LlamaCppRuntime):
    """Serves ternary models via PrismML's prebuilt llama.cpp."""

    kind = "prism"

    releases_api = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases?per_page=10"
    latest_api = "https://api.github.com/repos/PrismML-Eng/llama.cpp/releases/latest"
    bin_root = PRISM_BIN_ROOT
    label = "PrismML llama.cpp fork"
    cpu_patterns = (
        re.compile(r"bin-win-cpu-x64\.zip$", re.I),
        re.compile(r"bin-win-avx2-x64\.zip$", re.I),
    )

    def _tag_key(self, tag: str) -> tuple[int, int]:
        m = _TAG_RE.search(tag or "")
        return (1, int(m.group(1))) if m else (0, 0)

    def needs_build(self) -> bool:
        """False — the fork ships prebuilt Windows x64 CPU binaries."""
        return False

    def install_hint(self) -> str:
        return (
            f"Ternary models need {FORK_URL}, because stock llama.cpp cannot\n"
            "decode PQ2_0/PTQ1_0 packs. `llmlab setup <model>` downloads the\n"
            "prebuilt Windows x64 CPU binary automatically -- no CMake, no MSVC.\n"
            "To use a self-built binary instead, drop llama-server.exe in:\n"
            f"  {PRISM_BIN_ROOT}"
        )
