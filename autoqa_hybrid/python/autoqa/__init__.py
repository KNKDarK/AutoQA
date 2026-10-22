"""AutoQA — multimodal web QA agent with a Rust native engine.

The Python layer (`agent`, `driver`) drives the browser and the vision LLM;
the compiled Rust module `_native` provides fast screenshot diffing and DOM
mutation deltas.
"""
from autoqa.agent import AutoQAAgent
from autoqa.driver import BrowserDriver

try:  # pragma: no cover - depends on build environment
    from autoqa import _native as native  # noqa: F401  (re-exported)

    NATIVE_AVAILABLE = True
except ImportError:
    native = None
    NATIVE_AVAILABLE = False

__all__ = [
    "AutoQAAgent",
    "BrowserDriver",
    "native",
    "NATIVE_AVAILABLE",
    "__version__",
]
__version__ = "0.1.0"
