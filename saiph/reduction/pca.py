"""PCA projection module."""
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction.utils.common import (
    get_explained_variance,
    get_projected_column_names,
    get_uniform_row_weights,
)
from saiph.reduction.utils.svd import get_svd


def fit(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float_]] = None,
    seed: Optional[int] = None,
) -> Model:
    """Fit a PCA model on data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    nf = nf or min(df.shape)
    _col_weights = col_weights if col_weights is not None else np.ones(df.shape[1])

    # Set row weights
    row_w = get_uniform_row_weights(len(df))

    df_centered, mean, std = center(df)

    # Apply weights and compute svd
    Z = ((df_centered * _col_weights).T * row_w).T
    U, S, Vt = get_svd(Z, nf=nf, seed=seed)

    U = ((U.T) / np.sqrt(row_w)).T
    Vt = Vt / np.sqrt(_col_weights)

    explained_var, explained_var_ratio = get_explained_variance(S, df.shape[0], nf)

    # Retain only the nf higher singular values
    U = U[:, :nf]
    S = S[:nf]
    Vt = Vt[:nf, :]

    model = Model(
        original_dtypes=df.dtypes,
        original_categorical=[],
        original_continuous=df.columns.to_list(),
        dummy_categorical=[],
        U=U,
        V=Vt,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(Vt.T),
        mean=mean,
        std=std,
        type="pca",
        is_fitted=True,
        nf=nf,
        column_weights=_col_weights,
        row_weights=row_w,
        modalities_types={},
    )

    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_weights: Optional[NDArray[np.float_]] = None,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a PCA model on data and return transformed data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_weights: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
        coord: The transformed data.
    """
    model = fit(df, nf, col_weights, seed=seed)
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
    std[std <= sys.float_info.min] = 1
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
