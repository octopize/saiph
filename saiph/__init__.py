from .inverse_transform import inverse_transform
from .projection import fit, fit_transform, stats, transform

# Also modify in pyproject.toml
__version__ = "1.5.4"

__all__ = [
    "__version__",
    "fit",
    "fit_transform",
    "inverse_transform",
    "transform",
    "stats",
]
