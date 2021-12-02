from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
from numpy.typing import NDArray


@dataclass
class Model:
    df: pd.DataFrame
    """DataFrame on which the model was fit."""

    explained_var: NDArray[Any]
    """Explained variance."""
    explained_var_ratio: NDArray[Any]
    """Explained variance divided by the sum of the variances."""
    variable_coord: pd.DataFrame

    V: NDArray[Any]
    """Orthogonal matrix with right singular vectors as rows."""
    U: Optional[NDArray[Any]] = None
    """Orthogonal matrix with left singular vectors as columns."""
    s: Optional[NDArray[Any]] = None
    """Singular values"""

    mean: Optional[float] = None
    """Mean of the original data. Calculated while centering."""
    std: Optional[float] = None
    """Standard deviation of the original data. Calculated while scaling."""
    prop: Any = None  # FAMD only
    """Modality proportions of categorical variables"""
    _modalities: Optional[NDArray[Any]] = None
    D_c: Optional[NDArray[Any]] = None

    type: Optional[str] = None
    """Type of dimension reduction that was performed."""


@dataclass
class Parameters:
    nf: int
    """Number of components kept."""
    col_w: NDArray[Any]
    """Weights that were applied to each column."""
    row_w: NDArray[Any]
    """Weights that were applied to each row."""
    columns: List[Any]
    quanti: Optional[NDArray[Any]] = None
    """Indices of columns that are considered quantitative."""
    quali: Optional[NDArray[Any]] = None
    """Indices of columns that are considered qualitative."""
    datetime_variables: Optional[NDArray[Any]] = None
    """Indices of columns that are considered datetimes."""
    cor: Optional[pd.DataFrame] = None
    contrib: Optional[pd.DataFrame] = None
    cos2: Optional[pd.DataFrame] = None
    dummies_col_prop: Optional[NDArray[Any]] = None
    """Proportion of individuals taking each modality."""
