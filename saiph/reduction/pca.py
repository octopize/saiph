"""PCA projection."""
import sys
import typing

# from numpy.typing import ArrayLike
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model, Parameters
from saiph.reduction.utils.check_params import fit_check_params
from saiph.reduction.utils.common import (
    column_names,
    explain_variance,
    row_weights_uniform,
)
from saiph.reduction.utils.svd import SVD


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[Any]] = None,
    scale: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Model, Parameters]:
    """Project data into a lower dimensional space using PCA.

    Args:
        df: data to project
        nf: number of components to keep (default: {min(df.shape[0], 5)})
        col_w: importance of each variable in the projection
            (more weight = more importance in the axes)
        scale: wether to scale data or not

    Returns:
        The transformed variables, model and parameters
    """
    nf = nf or min(df.shape)
    _col_weights = col_weights or np.ones(df.shape[1])
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    fit_check_params(nf, _col_weights, df.shape[1])

    # set row weights
    row_w = row_weights_uniform(len(df))

    df_centered, mean, std = center(df, scale)

    # apply weights and compute svd
    Z = ((df_centered * _col_weights).T * row_w).T
    U, s, V = SVD(Z)

    U = ((U.T) / np.sqrt(row_w)).T
    V = V / np.sqrt(_col_weights)

    explained_var, explained_var_ratio = explain_variance(s, df_centered, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    columns = column_names(nf)

    coord = pd.DataFrame(np.dot(df_centered, V.T), columns=columns)  # type: ignore

    model = Model(
        df=df,
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(V.T),
        mean=mean,
        std=std,
    )

    param = Parameters(nf=nf, col_w=_col_weights, row_w=row_w, columns=columns)

    return coord, model, param


def center(
    df: pd.DataFrame, scale: Optional[bool] = True
) -> Tuple[pd.DataFrame, float, float]:
    """Center data. standardize data if scale == true. Compute mean and std."""
    df = df.copy()
    mean = np.mean(df, axis=0)
    df -= mean
    std = 0
    if scale:
        std = np.std(df, axis=0)
        std[std <= sys.float_info.min] = 1  # type: ignore
        df /= std
    return df, mean, std


def scaler(model: Model, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale data using mean and std from model."""
    if df is None:
        df = model.df

    df = df.copy()

    df -= model.mean
    df /= model.std
    return df


@typing.no_type_check
def transform(df: pd.DataFrame, model: Model, param: Parameters) -> pd.DataFrame:
    """Scale and project into the fitted numerical space."""
    df_scaled = scaler(model, df)
    return pd.DataFrame(np.dot(df_scaled, model.V.T), columns=param.columns)
