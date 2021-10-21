"""PCA projection."""
from saiph.reduction.utils.check_params import fit_check_params
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from saiph.models import Model, Parameters
from saiph.reduction.utils.bulk import column_names, explain_variance, row_weights_uniform
from saiph.reduction.utils.svd import SVD


def fit(
    _df: pd.DataFrame,
    nf: Optional[int] = None,
    col_w: Optional[ArrayLike] = None,
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
    nf = nf or min(_df.shape)
    col_w = col_w or np.ones(_df.shape[1])
    if not isinstance(_df, pd.DataFrame):
        _df = pd.DataFrame(_df)
    fit_check_params(nf, col_w, _df.shape[1])    
    df = _df.copy()
    df_original = df.copy()


    # df = np.array(df, copy=True, dtype="float64")

    # set row weights
    row_w = row_weights_uniform(len(df))

    df, mean, std = center(df, scale)

    # apply weights and compute svd
    Z = ((df * col_w).T * row_w).T
    U, s, V = SVD(Z)

    U = ((U.T) / np.sqrt(row_w)).T
    V = V / np.sqrt(col_w)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    columns = column_names(nf)

    coord = pd.DataFrame(np.dot(df, V.T), columns=columns)

    model = Model(
        df=df_original,
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(V.T),
        mean=mean,
        std=std,
    )

    param = Parameters(nf=nf, col_w=col_w, row_w=row_w, columns=columns)

    return coord, model, param


def center(
    df: pd.DataFrame, scale: Optional[bool] = True
) -> Tuple[pd.DataFrame, float, float]:
    """Center data. standardize data if scale == true. Compute mean and std."""
    mean = np.mean(df, axis=0)
    df -= mean
    std = 0
    if scale:
        std = np.std(df, axis=0)
        std[std <= sys.float_info.min] = 1
        df /= std
    return df, mean, std


def scaler(model: Model, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale data using mean and std."""
    if df is None:
        df = model.df

    df = df.copy()

    # df_scaled = np.array(df, copy=True, dtype="float64")

    # scale
    # df_scaled -= model.mean
    # df_scaled /= model.std
    df -= model.mean
    df /= model.std
    return df


def transform(df: pd.DataFrame, model: Model, param: Parameters) -> pd.DataFrame:
    """Scale and project into the fitted numerical space."""
    df_scaled = scaler(model, df)
    return pd.DataFrame(np.dot(df_scaled, model.V.T), columns=param.columns)
