"""Runtime registry.

Importing this package registers every available runtime.
"""

from .base import Runtime, RuntimeError_, ServerHandle, get_runtime, register_runtime
from . import llamacpp  # noqa: F401  (registers "llamacpp")
from . import prism  # noqa: F401  (registers "prism")

__all__ = [
    "Runtime",
    "RuntimeError_",
    "ServerHandle",
    "get_runtime",
    "register_runtime",
]
