from typing import Any, Optional, Tuple
import numpy as np

import pandas as pd
from numpy.typing import NDArray
from scipy import linalg
from sklearn.utils import extmath


def SVD(
    df: pd.DataFrame, svd_flip: bool = True, clip_value : Optional[np.float_] = None
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Parameters:
        df: Matrix to decompose.
        svd_flip: Whether to use svd_flip on U and V or not.

    Returns:
        U: Unitary matrix having left singular vectors as columns.
        s: The singular values.
        V: Unitary matrix having right singular vectors as rows.
    """
    U, s, V = linalg.svd(df, full_matrices=False)
    if svd_flip:
        U, V = extmath.svd_flip(U, V)

    if not clip_value:
        return U, s, V
    
    return clip(U, clip_value), clip(s, clip_value), clip(V, clip_value)

def clip(arr : NDArray[np.float_], clip_value : np.float_):
    
    _clip = np.abs(clip_value)
    _arr = arr.copy()
    _arr = np.where(np.logical_and(_arr > 0, _arr < _clip), _clip, _arr)
    _arr = np.where(np.logical_and(_arr < 0, _arr > -_clip), -_clip, _arr)

    return _arr
