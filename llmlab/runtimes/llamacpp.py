"""llama.cpp runtime (stock).

Uses the official prebuilt Windows binaries from ggml-org/llama.cpp releases,
so no compiler toolchain is required. Asset names have changed between release
lines, so we query the GitHub releases API and match by pattern rather than
hardcoding a filename.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

from ..manifest import Model
from ..registry import BIN_DIR, CACHE_DIR
from .base import Runtime, RuntimeError_, ServerHandle, register_runtime

RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20"
LATEST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
USER_AGENT = "local-llm-lab"

# Preference order for CPU builds, best first. Newer releases publish
# "cpu"; older ones used "avx2". Both are fine on an AVX2+ machine.
_CPU_PATTERNS = (
    re.compile(r"bin-win-cpu-x64\.zip$", re.I),
    re.compile(r"bin-win-avx2-x64\.zip$", re.I),
    re.compile(r"bin-win-cpu\.zip$", re.I),
)


def _api_get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@register_runtime
class LlamaCppRuntime(Runtime):
    kind = "llamacpp"

    # Subclasses (e.g. the PrismML fork) override these.
    releases_api = RELEASES_API
    latest_api = LATEST_API
    bin_root = BIN_DIR
    label = "llama.cpp"
    cpu_patterns = _CPU_PATTERNS

    def __init__(self) -> None:
        self.server_exe: Path | None = None

    # -- install -----------------------------------------------------------
    def _pick_asset(self, assets: list[dict]) -> dict:
        for pattern in self.cpu_patterns:
            for asset in assets:
                if pattern.search(asset.get("name", "")):
                    return asset
        names = [a.get("name", "") for a in assets]
        raise RuntimeError_(
            f"no Windows x64 CPU binary found in the latest {self.label} release. "
            f"Available assets: {', '.join(names) or '(none)'}"
        )

    def _tag_key(self, tag: str) -> tuple[int, int]:
        """Order releases newest-first. Default matches rolling bNNNNN tags."""
        m = re.fullmatch(r"b(\d+)", tag)
        return (1, int(m.group(1))) if m else (0, 0)

    def _find_server(self) -> Path | None:
        if not self.bin_root.is_dir():
            return None
        for candidate in self.bin_root.rglob("llama-server.exe"):
            return candidate
        return None

    def _fetch_release(self) -> dict:
        """Find a release that actually carries Windows binaries.

        `/releases/latest` on ggml-org/llama.cpp points at a stable tag whose
        only asset is `nightly-tag.txt`. The real prebuilt binaries live on
        the rolling `bNNNNN` prereleases, so we scan the release list for the
        newest one that has a matching asset.
        """
        try:
            releases = _api_get(self.releases_api)
        except Exception:
            releases = []

        if isinstance(releases, list):
            ordered = sorted(
                releases,
                key=lambda rel: self._tag_key(rel.get("tag_name") or ""),
                reverse=True,
            )
            for rel in ordered:
                for pattern in self.cpu_patterns:
                    if any(pattern.search(a.get("name", "")) for a in rel.get("assets", [])):
                        return rel

        # Fall back to the stable "latest" release.
        return _api_get(self.latest_api)

    def ensure_installed(self) -> None:
        existing = self._find_server()
        if existing:
            self.server_exe = existing
            return

        self.bin_root.mkdir(parents=True, exist_ok=True)
        print(f"llmlab: resolving latest {self.label} Windows build...")
        release = self._fetch_release()
        tag = release.get("tag_name", "?")
        asset = self._pick_asset(release.get("assets", []))
        size_mb = asset.get("size", 0) / 1_048_576
        print(f"llmlab: downloading {tag} / {asset['name']} ({size_mb:.0f} MB)...")

        req = urllib.request.Request(
            asset["browser_download_url"], headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            blob = r.read()

        print("llmlab: extracting...")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(self.bin_root)

        found = self._find_server()
        if not found:
            raise RuntimeError_(
                f"extracted archive but could not find llama-server.exe under {self.bin_root}"
            )
        self.server_exe = found
        print(f"llmlab: llama-server ready at {found}")

    @property
    def version(self) -> str | None:
        """Best-effort version string from the binary.

        llama-server --version writes init noise to stderr alongside the real
        version line, so pick the first line that actually looks like a version.
        """
        if not self.server_exe or not self.server_exe.exists():
            return None
        try:
            out = subprocess.run(
                [str(self.server_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            return None
        blob = (out.stdout or "") + "\n" + (out.stderr or "")
        for line in blob.splitlines():
            if re.search(r"\bversion\b", line, re.I) and re.search(r"\d", line):
                return line.strip()
        return None

    # -- lifecycle ---------------------------------------------------------
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

        ctx = ctx_size or model.runtime.ctx_size
        threads = model.runtime.threads or (os.cpu_count() or 8)

        cmd = [
            str(self.server_exe),
            "-m", str(model.gguf_path),
            "-c", str(ctx),
            "-t", str(threads),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--alias", model.id,
        ]
        cmd.extend(model.runtime.extra_args)

        log_dir = CACHE_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{model.id}-{port}.log"
        log_file = open(log_path, "wb")

        print(
            f"llmlab: starting {model.id} on port {port} "
            f"(ctx={ctx}, threads={threads})..."
        )
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            log_file.close()
            raise RuntimeError_(f"failed to launch llama-server: {exc}") from exc

        return ServerHandle(port=port, process=proc, log_path=log_path)

    def stop(self, handle: ServerHandle) -> None:
        proc = handle.process
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            if handle.log_path:
                try:
                    handle.log_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

    def tail_log(self, handle: ServerHandle, lines: int = 25) -> str:
        if not handle.log_path or not handle.log_path.exists():
            return "(no log)"
        text = handle.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])
