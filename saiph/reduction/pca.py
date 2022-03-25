"""PCA projection module."""
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction.utils.check_params import fit_check_params
from saiph.reduction.utils.common import (
    explain_variance,
    get_projected_column_names,
    get_uniform_row_weights,
)
from saiph.reduction.utils.svd import SVD


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Model:
    """Fit a PCA model on data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_w: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    nf = nf or min(df.shape)
    if col_w is not None:
        _col_weights = col_w
    else:
        _col_weights = np.ones(df.shape[1])

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    fit_check_params(nf, _col_weights, df.shape[1])

    # set row weights
    row_w = get_uniform_row_weights(len(df))

    df_centered, mean, std = center(df)

    # apply weights and compute svd
    Z = ((df_centered * _col_weights).T * row_w).T
    U, s, V = SVD(Z)

    U = ((U.T) / np.sqrt(row_w)).T
    V = V / np.sqrt(_col_weights)

    explained_var, explained_var_ratio = explain_variance(s, df_centered, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    model = Model(
        original_dtypes=df.dtypes,
        original_categorical=[],
        original_continuous=df.columns.to_list(),
        dummy_categorical=[],
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(V.T),
        mean=mean,
        std=std,
        type="pca",
        is_fitted=True,
        nf=nf,
        column_weights=_col_weights,
        row_weights=row_w,
    )

    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a PCA model on data and return transformed data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_w: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
        coord: The transformed data.
    """
    model = fit(df, nf, col_w)
    coord = transform(df, model)
    return coord, model


def center(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Center data and standardize it if scale. Compute mean and std values.

    Used as internal function during fit.

    **NB**: saiph.reduction.pca.scaler is better suited when a Model is already fitted.

    Parameters:
        df: DataFrame to center.

    Returns:
        df: The centered DataFrame.
        mean: Mean of the input dataframe.
        std: Standard deviation of the input dataframe.
    """
    df = df.copy()
    mean = np.mean(df, axis=0)
    df -= mean

    std = np.std(df, axis=0)
    std[std <= sys.float_info.min] = 1  # type: ignore
    df /= std

    return df, mean, std


def scaler(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using mean and std from model.

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df: The scaled DataFrame.
    """
    df_scaled = df.copy()

    df_scaled -= model.mean
    df_scaled /= model.std
    return df_scaled


def transform(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space.
    """
    df_scaled = scaler(model, df)
    coord = df_scaled @ model.V.T
    coord.columns = get_projected_column_names(model.nf)
    return coord
