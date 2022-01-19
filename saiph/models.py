from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Model:
    df: pd.DataFrame
    """DataFrame on which the model was fit."""

    explained_var: NDArray[np.float_]
    """Explained variance."""
    explained_var_ratio: NDArray[np.float_]
    """Explained variance divided by the sum of the variances."""
    variable_coord: pd.DataFrame
    """Coordinates of the variables in the projection space."""
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
    """Modality proportions of categorical variables."""
    _modalities: Optional[NDArray[Any]] = None
    """Modalities for the MCA/FAMD."""
    D_c: Optional[NDArray[Any]] = None
    """Diagonal matrix containing sums along columns of the scaled data as diagonals."""
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
    """Column names once data is projected."""
    quanti: Optional[NDArray[Any]] = None
    """Indices of columns that are considered quantitative."""
    quali: Optional[NDArray[Any]] = None
    """Indices of columns that are considered qualitative."""
    datetime_variables: Optional[NDArray[Any]] = None
    """Indices of columns that are considered datetimes."""
    cor: Optional[pd.DataFrame] = None
    """Correlation between the axis and the variables."""
    contrib: Optional[pd.DataFrame] = None
    """Contributions for each variable."""
    cos2: Optional[pd.DataFrame] = None
    """Cos2 for each variable."""
    dummies_col_prop: Optional[NDArray[Any]] = None
    """Proportion of individuals taking each modality."""
