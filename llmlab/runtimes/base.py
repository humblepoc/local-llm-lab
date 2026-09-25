"""Runtime abstraction.

A Runtime knows how to make a model *serve* and how to talk to it. Every
runtime exposes the same OpenAI-compatible surface, so the bench and eval
harnesses never care which one is underneath.
"""

from __future__ import annotations

import abc
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..manifest import Model


@dataclass
class ServerHandle:
    """A running server, plus whatever the runtime needs to stop it."""

    port: int
    process: Any = None
    started_at: float = field(default_factory=time.time)
    log_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class RuntimeError_(RuntimeError):
    pass


class Runtime(abc.ABC):
    """Base class for serving backends."""

    kind: str = "abstract"

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    def ensure_installed(self) -> None:
        """Make the server binary available. Idempotent."""

    @abc.abstractmethod
    def start(self, model: Model, port: int, ctx_size: int | None = None) -> ServerHandle:
        """Launch a server for ``model``."""

    @abc.abstractmethod
    def stop(self, handle: ServerHandle) -> None:
        """Terminate a running server and clean up."""

    def needs_build(self) -> bool:
        """True if this runtime requires a from-source toolchain."""
        return False

    def install_hint(self) -> str:
        return ""

    # -- introspection -----------------------------------------------------
    def health(self, handle: ServerHandle) -> bool:
        try:
            with urllib.request.urlopen(f"{handle.base_url}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_ready(self, handle: ServerHandle, timeout: float = 300.0) -> bool:
        """Poll /health until the server answers or the timeout expires.

        Model load time on CPU can be substantial, so this is generous by
        default and the caller can raise it for very large models.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if handle.process is not None and handle.process.poll() is not None:
                return False
            if self.health(handle):
                return True
            time.sleep(1.0)
        return False

    # -- inference ---------------------------------------------------------
    def chat(
        self,
        handle: ServerHandle,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Non-streaming OpenAI-compatible chat completion."""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        req = urllib.request.Request(
            f"{handle.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError_(f"HTTP {exc.code} from {handle.base_url}: {body}") from exc

    def chat_stream(
        self,
        handle: ServerHandle,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ):
        """Yield (elapsed_seconds, chunk_dict) so callers can time TTFT."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{handle.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    return
                try:
                    yield time.time() - start, json.loads(payload_str)
                except json.JSONDecodeError:
                    continue


_REGISTRY: dict[str, type[Runtime]] = {}


def register_runtime(cls: type[Runtime]) -> type[Runtime]:
    _REGISTRY[cls.kind] = cls
    return cls


def get_runtime(kind: str) -> Runtime:
    try:
        return _REGISTRY[kind]()
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown runtime kind '{kind}'. known: {known}") from None
