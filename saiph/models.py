from dataclasses import dataclass
from typing import Any, Optional, Union

import pandas as pd
from numpy.typing import ArrayLike, NDArray

DFLike = Union[pd.DataFrame, ArrayLike]


@dataclass
class Model:
    df: pd.DataFrame

    explained_var: ArrayLike
    explained_var_ratio: ArrayLike
    variable_coord: DFLike

    # Orthogonal matrix with right singular vectors as rows
    V: NDArray[Any]
    # Orthogonal matrix with left singular vectors as columns
    U: Optional[NDArray[Any]] = None
    # Singular values
    s: Optional[NDArray[Any]] = None

    mean: Optional[float] = None
    std: Optional[float] = None
    prop: Optional[float] = None

    _modalities: Optional[NDArray[Any]] = None
    D_c: Optional[ArrayLike] = None


@dataclass
class Parameters:
    nf: int
    col_w: NDArray[Any]
    row_w: ArrayLike
    columns: ArrayLike
    quanti: Optional[NDArray[Any]] = None
    quali: Optional[NDArray[Any]] = None
    datetime_variables: Optional[NDArray[Any]] = None
    cor: Optional[DFLike] = None
    contrib: Optional[DFLike] = None
    cos2: Optional[DFLike] = None
