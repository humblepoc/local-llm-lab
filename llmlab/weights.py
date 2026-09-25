"""Weight acquisition.

Weights are never committed to this repository. Each model manifest points at a
Hugging Face repo; ``llmlab setup`` fetches the file, resumes if interrupted, and
writes a ``.sha256`` sidecar so later runs can verify integrity.

Two download backends, chosen automatically:

* **huggingface_hub + hf_transfer** - parallel chunked download, typically far
  faster on HF. Optional; used when both are importable.
* **urllib** - the default zero-dependency fallback, with HTTP Range resume.

Set ``LLMLAB_DISABLE_HF_TRANSFER=1`` to force the urllib path (useful for
debugging, or on networks where the parallel fetcher misbehaves).
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .manifest import Model

USER_AGENT = "local-llm-lab"
CHUNK = 1024 * 1024


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model: Model) -> tuple[bool, str]:
    """Compare on-disk weight against the manifest checksum."""
    if not model.source.sha256:
        return True, "no checksum declared in manifest (skipped)"
    if not model.installed:
        return False, "weights not downloaded"
    actual = sha256_file(model.gguf_path)
    expected = model.source.sha256.strip().lower()
    ok = actual == expected
    if ok:
        return True, f"sha256 ok ({actual[:16]}...)"
    return False, f"sha256 MISMATCH\n      expected {expected}\n      actual   {actual}"


def _hf_major() -> int | None:
    """Major version of the installed huggingface_hub, or None if absent."""
    try:
        from huggingface_hub import __version__
    except Exception:
        return None
    try:
        return int(str(__version__).split(".")[0])
    except ValueError:
        return None


def hf_transfer_status() -> tuple[bool, str]:
    """Report whether the accelerated backend is usable, and why not if it isn't."""
    if os.environ.get("LLMLAB_DISABLE_HF_TRANSFER") == "1":
        return False, "disabled via LLMLAB_DISABLE_HF_TRANSFER=1"
    major = _hf_major()
    if major is None:
        return False, "huggingface_hub not installed - using urllib"
    if major >= 1:
        # huggingface_hub >=1.0 replaced hf_transfer with the Xet backend and
        # deprecated HF_HUB_ENABLE_HF_TRANSFER.
        return True, f"available (huggingface_hub {major}.x, Xet backend)"
    try:
        __import__("hf_transfer")
    except ImportError:
        return False, "hf_transfer not installed (needed for huggingface_hub <1.0)"
    return True, "available (hf_transfer)"


def _fetch_via_hf_transfer(model: Model) -> bool:
    """Download via huggingface_hub + hf_transfer. Returns True on success.

    Falls back (returns False) on any problem so the urllib path can take over.
    Downloads straight into the weights dir via ``local_dir`` so the resulting
    layout is identical to the urllib backend.
    """
    ok, why = hf_transfer_status()
    if not ok:
        print(f"llmlab: {why}")
        return False

    major = _hf_major() or 0
    if major >= 1:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
        print("llmlab: using huggingface_hub (Xet backend, high performance)")
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("llmlab: using hf_transfer (parallel chunked download)")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - import already checked
        print(f"llmlab: huggingface_hub import failed ({exc}); using urllib")
        return False

    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=model.source.repo,
            filename=model.source.file,
            local_dir=str(model.weights_dir),
        )
    except Exception as exc:
        print(f"llmlab: hf_transfer download failed ({type(exc).__name__}: {exc})")
        print("llmlab: falling back to urllib")
        return False

    got = Path(path)
    # Defensive: some hub versions place the file under a cache subdir.
    if got.resolve() != model.gguf_path.resolve():
        model.weights_dir.mkdir(parents=True, exist_ok=True)
        got.replace(model.gguf_path)
    took = time.time() - t0
    size_mb = model.gguf_path.stat().st_size / 1_048_576
    print(
        f"llmlab: fetched {size_mb:.0f} MB in {took:.0f}s "
        f"({size_mb/max(took,1e-9):.1f} MB/s)"
    )
    return True


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Download with Range-based resume. Writes to ``dest.part`` until complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:  # already complete
            partial.replace(dest)
            return
        raise

    mode = "ab" if existing and resp.status == 206 else "wb"
    if mode == "wb":
        existing = 0

    total = resp.headers.get("Content-Length")
    total_i = int(total) + existing if total and total.isdigit() else None

    done = existing
    reported = existing
    last_report = 0.0
    with open(partial, mode) as out:
        while True:
            block = resp.read(CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            now = time.time()
            if now - last_report > 2.0:
                if total_i:
                    pct = 100 * done / total_i
                    interval = max(now - last_report, 1e-9)
                    rate = (done - reported) / interval
                    print(
                        f"\r  {pct:5.1f}%  {_human(done)}/{_human(total_i)}"
                        f"  {_human(rate)}/s   ",
                        end="",
                        flush=True,
                    )
                    reported = done
                else:
                    print(f"\r  {_human(done)}   ", end="", flush=True)
                last_report = now
    print()
    partial.replace(dest)


def fetch(model: Model, force: bool = False) -> Path:
    """Ensure weights are present and verified. Returns the weights path."""
    if model.installed and not force:
        ok, msg = verify(model)
        if ok:
            print(f"llmlab: {model.id} already present ({_human(model.gguf_path.stat().st_size)})")
            print(f"        {msg}")
            return model.gguf_path
        if not force:
            print(f"llmlab: re-downloading {model.id} — {msg}")

    model.weights_dir.mkdir(parents=True, exist_ok=True)
    url = model.source.url
    expected = model.source.size_gb

    if not _fetch_via_hf_transfer(model):
        if expected:
            print(f"llmlab: downloading {model.id} ({expected} GB expected)")
        else:
            print(f"llmlab: downloading {model.id}")
        print(f"        {url}")
        _download(url, model.gguf_path)

    size = model.gguf_path.stat().st_size
    print(f"llmlab: downloaded {_human(size)}")

    ok, msg = verify(model)
    print(f"llmlab: {msg}")
    if not ok:
        raise RuntimeError(
            f"checksum verification failed for {model.id}.\n"
            f"  Delete {model.gguf_path} and re-run, or check the manifest checksum."
        )

    sidecar = model.gguf_path.with_suffix(model.gguf_path.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256_file(model.gguf_path)}  {model.gguf_path.name}\n", encoding="utf-8"
    )
    return model.gguf_path
