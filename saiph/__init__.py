from .inverse_transform import inverse_transform
from .projection import fit, fit_transform, stats, transform
from .streaming import (
    DecompositionAccumulator,
    ScalingAccumulator,
    ScalingParams,
    fit_streaming,
)

# Also modify in pyproject.toml
__version__ = "3.0.0"

__all__ = [
    "DecompositionAccumulator",
    "ScalingAccumulator",
    "ScalingParams",
    "__version__",
    "fit",
    "fit_streaming",
    "fit_transform",
    "inverse_transform",
    "stats",
    "transform",
]
