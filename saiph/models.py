from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

ListLike = Union[np.array, list]
DFLike = Union[pd.DataFrame, np.array]


@dataclass
class Model:
    df: DFLike
    V: ListLike
    explained_var: ListLike
    explained_var_ratio: ListLike
    variable_coord: DFLike
    U: Optional[ListLike] = None
    s: Optional[ListLike] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    prop: Optional[float] = None
    _modalities: Optional[np.array] = None
    D_c: Optional[np.array] = None


@dataclass
class Parameters:
    nf: int
    col_w: ListLike
    row_w: ListLike
    columns: list
    quanti: Optional[ListLike] = None
    quali: Optional[ListLike] = None
    datetime_variables: Optional[list] = field(default_factory=list)
    cor: Optional[DFLike] = None
    contrib: Optional[DFLike] = None
    cos2: Optional[DFLike] = None
