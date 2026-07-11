"""cognate_reflexes — extract training examples from Lexibank for LLM-based
proto-language reconstruction.

Public API
----------
Config              Central pipeline configuration.
ExampleGenerator    Builds cognate-reflex / reconstruction examples.
TextFormatter       Serialises examples into model-ready text.
split_by_family     Splits examples into train/dev/test by language family.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover – resolved at type-check time only
    from cognate_reflexes.config import Config as Config
    from cognate_reflexes.formatting.formatter import TextFormatter as TextFormatter
    from cognate_reflexes.splitting.family_split import split_by_family as split_by_family
    from cognate_reflexes.examples.generator import ExampleGenerator as ExampleGenerator

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ExampleGenerator",
    "TextFormatter",
    "split_by_family",
]

# ---------------------------------------------------------------------------
# Lazy imports – modules are loaded only when their names are first accessed.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Config": ("cognate_reflexes.config", "Config"),
    "ExampleGenerator": ("cognate_reflexes.examples.generator", "ExampleGenerator"),
    "TextFormatter": ("cognate_reflexes.formatting.formatter", "TextFormatter"),
    "split_by_family": ("cognate_reflexes.splitting.family_split", "split_by_family"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

