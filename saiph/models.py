from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class Model:
    # List of categorical columns transformed into dummies using pd.get_dummies
    dummy_categorical: list[str]

    # List of original columns with dtypes generated with df.dtypes.
    # Calling .index refers to column names,
    # Calling .values refers to the dtypes of the column names.
    original_dtypes: pd.Series

    # Original categorical column names
    original_categorical: list[str]
    # Original continuous column names
    original_continuous: list[str]

    # Number of components kept.
    nf: int
    # Weights that were applied to each column.
    column_weights: NDArray[np.float64]
    # Weights that were applied to each row.
    row_weights: NDArray[np.float64]

    # Explained variance.
    explained_var: NDArray[np.float64]
    # Explained variance divided by the sum of the variances.
    explained_var_ratio: NDArray[np.float64]
    # Coordinates of the variables in the projection space.
    variable_coord: pd.DataFrame
    # Orthogonal matrix with right singular vectors as rows.
    V: NDArray[np.float64]
    # Modality type of the first individuals
    modalities_types: dict[str, str]
    # Orthogonal matrix with left singular vectors as columns.
    U: NDArray[np.float64]
    # Singular values
    s: NDArray[np.float64] | None = None

    # Mean of the original data. Calculated while centering.
    mean: pd.Series | None = None
    # Standard deviation of the original data. Calculated while scaling.
    std: pd.Series | None = None

    # Modality proportions of categorical variables.
    prop: pd.Series | None = None  # FAMD only
    # Modalities for the MCA/FAMD.
    _modalities: NDArray[np.bytes_] | None = None
    # Diagonal matrix containing sums along columns of the scaled data as diagonals.
    D_c: NDArray[np.float64] | None = None
    # Type of dimension reduction that was performed.
    type: str | None = None

    is_fitted: bool = False

    # Correlation between the axis and the variables.
    correlations: pd.DataFrame | None = None
    # Contributions for each variable.
    contributions: pd.DataFrame | None = None
    # Cos2 for each variable.
    cos2: pd.DataFrame | None = None
    # Proportion of individuals taking each modality.
    dummies_col_prop: NDArray[np.float64] | None = None  # MCA only

    seed: int | None = None
