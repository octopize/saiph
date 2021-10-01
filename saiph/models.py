from dataclasses import dataclass, field
from typing import Optional, Union

from numpy.typing import ArrayLike
import pandas as pd

DFLike = Union[pd.DataFrame, ArrayLike]


@dataclass
class Model:
    df: DFLike
    V: ArrayLike
    explained_var: ArrayLike
    explained_var_ratio: ArrayLike
    variable_coord: DFLike
    U: Optional[ArrayLike] = None
    s: Optional[ArrayLike] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    prop: Optional[float] = None
    _modalities: Optional[ArrayLike] = None
    D_c: Optional[ArrayLike] = None


@dataclass
class Parameters:
    nf: int
    col_w: ArrayLike
    row_w: ArrayLike
    columns: ArrayLike
    quanti: Optional[ArrayLike] = None
    quali: Optional[ArrayLike] = None
    datetime_variables: Optional[ArrayLike] = field(default_factory=list)  # type: ignore
    cor: Optional[DFLike] = None
    contrib: Optional[DFLike] = None
    cos2: Optional[DFLike] = None
