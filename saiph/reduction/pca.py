"""PCA projection."""
import sys
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
    col_w: Optional[NDArray[Any]] = None,
    scale: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Model, Parameters]:
    """Fit a PCA model on data.

    Parameters
    ----------
    df: pd.DataFrame
        Data to project.
    nf: int, default: min(df.shape)
        Number of components to keep.
    col_w: np.ndarrayn default: np.ones(df.shape[1])
        Weight assigned to each variable in the projection
        (more weight = more importance in the axes).
    scale: bool
        Wether to scale data or not.

    Returns
    -------
    coord: pd.DataFrame
        The transformed data.
    model: Model
        The model for transforming new data.
    param: Parameters
        The parameters for transforming new data.
    """
    nf = nf or min(df.shape)
    _col_weights = col_w or np.ones(df.shape[1])
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
    coord = df_centered @ V.T
    coord.columns = columns

    model = Model(
        df=df,
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(V.T),
        mean=mean,
        std=std,
        type="pca",
    )

    param = Parameters(nf=nf, col_w=_col_weights, row_w=row_w, columns=columns)

    return coord, model, param


def center(
    df: pd.DataFrame, scale: Optional[bool] = True
) -> Tuple[pd.DataFrame, float, float]:
    """Center data and standardize it if scale == true.

    Used as internal function during fit. Scaler is better suited when a Model is already fitted.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to center.
    scale: bool
        Whether dataframe should be standardized ot not.

    Returns
    -------
    df: pd.DataFrame
        The centered DataFrame.
    mean: float
        Mean of the input dataframe.
    std: float
        Standard deviation of the input dataframe. Returns nan as std if no std was asked.
    """
    df = df.copy()
    mean = np.mean(df, axis=0)
    df -= mean
    std = 0
    if scale:
        std = np.std(df, axis=0)
        std[std <= sys.float_info.min] = 1  # type: ignore
        df /= std
    # else:
    #     std = np.nan
    return df, mean, std


def scaler(model: Model, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale data using mean and std from model.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    df: pd.DataFrame
        DataFrame to scale.
        If nothing is specified, takes the DataFrame on which the model was fitted.

    Returns
    -------
    df: pd.DataFrame
        The scaled DataFrame.
    """
    if df is None:
        df = model.df

    df_scaled = df.copy()

    df_scaled -= model.mean
    df_scaled /= model.std
    return df_scaled


def transform(df: pd.DataFrame, model: Model, param: Parameters) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to transform.
    model: Model
        Model computed by fit.
    param: Parameters
        Param computed by fit.

    Returns
    -------
    coord: pd.DataFrame
        Coordinates of the dataframe in the fitted space.
    """
    df_scaled = scaler(model, df)
    coord = df_scaled @ model.V.T
    coord.columns = param.columns
    return coord
