from .projection import fit, inverse_transform, stats, transform
from .visualization import plot_circle, plot_explained_var, plot_var_contribution

# Also modify in pyproject.toml
__version__ = "0.2.5"

__all__ = [
    "__version__",
    "fit",
    "inverse_transform",
    "transform",
    "stats",
    "plot_circle",
    "plot_explained_var",
    "plot_var_contribution",
]
