"""PCA projection."""
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from saiph.models import Model, Parameters
from saiph.svd import SVD


def fit(
    df: pd.DataFrame,
    nf: int = None,
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
    if not nf:
        nf = min(df.shape)
    elif nf <= 0:
        raise ValueError("nf", "The number of components must be positive.")

    if not col_w:
        col_w = np.ones(df.shape[1])
    elif len(col_w) != df.shape[1]:
        raise ValueError(
            "col_w",
            f"The weight parameter should be of size {str(df.shape[1])}.",
        )

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df_original = df.copy()
    df = np.array(df, copy=True, dtype="float64")

    # set row weights
    row_w = [1 / len(df) for i in range(len(df))]

    df, mean, std = center(df, scale)

    # apply weights and compute svd
    Z = ((df * col_w).T * row_w).T
    U, s, V = SVD(Z)

    U = ((U.T) / np.sqrt(row_w)).T
    V = V / np.sqrt(col_w)

    # compute eigenvalues and explained variance
    explained_var = ((s ** 2) / (df.shape[0] - 1))[:nf]  # type: ignore
    summed_explained_var = explained_var.sum()
    if summed_explained_var == 0:
        explained_var_ratio = np.nan
    else:
        explained_var_ratio = explained_var / explained_var.sum()

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    columns = [f"Dim. {i}" for i in range(min(nf, len(df)))]

    coord = pd.DataFrame(np.dot(df, V.T), columns=columns)
    model = Model(
        df=df_original,
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


def scaler(model: Model, df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Scale data using mean and std."""
    if df is None:
        df = model.df

    df_scaled = np.array(df, copy=True, dtype="float64")

    # scale
    df_scaled -= model.mean
    df_scaled /= model.std
    return df_scaled


def transform(df: pd.DataFrame, model: Model, param: Parameters) -> pd.DataFrame:
    """Scale and Project new data."""
    df_scaled = scaler(model, df)
    return pd.DataFrame(np.dot(df_scaled, model.V.T), columns=param.columns)
