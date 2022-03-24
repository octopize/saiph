from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Model:

    # List of categorical columns transformed into dummies using pd.get_dummies
    dummy_categorical: List[str]

    # List of original columns with dtypes generated with df.dtypes.
    # Calling .index refers to column names,
    # Calling .values refers to the dtypes of the column names.
    original_dtypes: pd.Series

    # Original categorical column names
    original_categorical: List[str]
    # Original continuous column names
    original_continuous: List[str]

    # Number of components kept.
    nf: int
    # Weights that were applied to each column.
    column_weights: NDArray[np.float_]
    # Weights that were applied to each row.
    row_weights: NDArray[np.float_]

    # Explained variance.
    explained_var: NDArray[np.float_]
    # Explained variance divided by the sum of the variances.
    explained_var_ratio: NDArray[np.float_]
    # Coordinates of the variables in the projection space.
    variable_coord: pd.DataFrame
    # Orthogonal matrix with right singular vectors as rows.
    V: NDArray[np.float_]
    # Orthogonal matrix with left singular vectors as columns.
    U: Optional[NDArray[np.float_]] = None
    # Singular values
    s: Optional[NDArray[np.float_]] = None

    # Mean of the original data. Calculated while centering.
    mean: Optional[pd.Series] = None
    # Standard deviation of the original data. Calculated while scaling.
    std: Optional[pd.Series] = None

    # Modality proportions of categorical variables.
    prop: Any = None  # FAMD only
    # Modalities for the MCA/FAMD.
    _modalities: Optional[NDArray[Any]] = None
    # Diagonal matrix containing sums along columns of the scaled data as diagonals.
    D_c: Optional[NDArray[Any]] = None
    # Type of dimension reduction that was performed.
    type: Optional[str] = None

    is_fitted: bool = False

    # Correlation between the axis and the variables.
    correlations: Optional[pd.DataFrame] = None
    # Contributions for each variable.
    contributions: Optional[pd.DataFrame] = None
    # Cos2 for each variable.
    cos2: Optional[pd.DataFrame] = None
    # Proportion of individuals taking each modality.
    dummies_col_prop: Optional[NDArray[np.float_]] = None
