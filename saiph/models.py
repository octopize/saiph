from dataclasses import dataclass
from typing import Any, List, Optional, Union

import pandas as pd
from numpy.typing import NDArray


@dataclass
class Model:
    df: pd.DataFrame

    explained_var: NDArray[Any]
    explained_var_ratio: Union[NDArray[Any], float]
    variable_coord: pd.DataFrame

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
    D_c: Optional[NDArray[Any]] = None


@dataclass
class Parameters:
    nf: int
    col_w: NDArray[Any]
    row_w: NDArray[Any]
    columns: List[Any]
    quanti: Optional[NDArray[Any]] = None
    quali: Optional[NDArray[Any]] = None
    datetime_variables: Optional[NDArray[Any]] = None
    cor: Optional[pd.DataFrame] = None
    contrib: Optional[pd.DataFrame] = None
    cos2: Optional[pd.DataFrame] = None
