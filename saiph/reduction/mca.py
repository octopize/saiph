"""MCA projection module."""
from itertools import chain, repeat
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.utils.check_params import fit_check_params
from saiph.reduction.utils.common import (
    diag,
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
    """Fit a MCA model on data.

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

    # initiate row and columns weights
    row_weights = get_uniform_row_weights(len(df))

    modality_numbers = []
    for column in df.columns:
        modality_numbers += [len(df[column].unique())]
    col_weights: NDArray[Any] = np.array(
        list(
            chain.from_iterable(
                repeat(i, j) for i, j in zip(_col_weights, modality_numbers)
            )
        )
    )

    df_scale, _modalities, r, c = center(df)
    df_scale, T, D_c = _diag_compute(df_scale, r, c)

    # get the array gathering proportion of each modality among individual (N/n)
    df_dummies = pd.get_dummies(df.astype("category"), prefix_sep=DUMMIES_PREFIX_SEP)
    dummies_col_prop = len(df_dummies) / df_dummies.sum(axis=0)

    # apply the weights and compute the svd
    Z = ((T * col_weights).T * row_weights).T
    U, s, V = SVD(Z)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    model = Model(
        original_dtypes=df.dtypes,
        original_categorical=df.columns.to_list(),
        original_continuous=[],
        dummy_categorical=df_dummies.columns.to_list(),
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=D_c @ V.T,
        _modalities=_modalities,
        D_c=D_c,
        type="mca",
        is_fitted=True,
        nf=nf,
        column_weights=col_weights,
        row_weights=row_weights,
        dummies_col_prop=dummies_col_prop,
    )

    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a MCA model on data and return transformed data.

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


def center(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Center data and compute modalities.

    Used as internal function during fit.

    **NB**: saiph.reduction.mca.scaler is better suited when a Model is already fitted.

    Parameters:
        df: DataFrame to center.

    Returns:
        df_centered: The centered DataFrame.
        _modalities: Modalities for the MCA
        row_sum: Sums line by line
        column_sum: Sums column by column
    """
    df_scale = pd.get_dummies(df.astype("category"), prefix_sep=DUMMIES_PREFIX_SEP)
    _modalities = df_scale.columns.values

    # scale data
    df_scale /= df_scale.sum().sum()

    row_sum = np.sum(df_scale, axis=1)
    column_sum = np.sum(df_scale, axis=0)
    return df_scale, _modalities, row_sum, column_sum


def scaler(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using modalities from model.

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df_scaled: The scaled DataFrame.
    """
    df_scaled = pd.get_dummies(df.astype("category"), prefix_sep=DUMMIES_PREFIX_SEP)
    if model._modalities is not None:
        for mod in model._modalities:
            if mod not in df_scaled:
                df_scaled[mod] = 0
    df_scaled = df_scaled[model._modalities]

    # scale
    df_scaled /= df_scaled.sum().sum()
    df_scaled /= np.array(np.sum(df_scaled, axis=1))[:, None]
    return df_scaled


def _diag_compute(
    df_scale: pd.DataFrame, r: NDArray[Any], c: NDArray[Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute diagonal matrices and scale data."""
    eps: np.float_ = np.finfo(float).eps
    if df_scale.shape[0] >= 10000:
        D_r = diag(1 / (eps + np.sqrt(r)), use_scipy=True)
    else:
        D_r = diag(1 / (eps + np.sqrt(r)), use_scipy=False)
    D_c = diag(1 / (eps + np.sqrt(c)), use_scipy=False)

    T = D_r @ (df_scale - np.outer(r, c)) @ D_c
    return df_scale / np.array(r)[:, None], T, D_c


def transform(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space.
    """
    df_scaled = scaler(model, df)
    coord = df_scaled @ model.D_c @ model.V.T
    coord.columns = get_projected_column_names(model.nf)
    return coord


def get_variable_contributions(model: Model, df: pd.DataFrame) -> NDArray[np.float_]:
    """Compute the contributions of the `df` variables within the fitted space.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute contributions from

    Returns:
        contributions
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    V = np.dot(model.D_c, model.V.T)  # type: ignore
    df = pd.get_dummies(df.astype("category"), prefix_sep=DUMMIES_PREFIX_SEP)
    F = df / df.sum().sum()

    # Column and row weights
    marge_col = F.sum(axis=0)
    marge_row = F.sum(axis=1)
    fsurmargerow = _rdivision(F, marge_row)
    fmargerowT = pd.DataFrame(
        np.array(fsurmargerow).T,
        columns=list(fsurmargerow.index),
        index=list(fsurmargerow.columns),
    )
    fmargecol = _rdivision(fmargerowT, marge_col)
    Tc = (
        pd.DataFrame(
            np.array(fmargecol).T,
            columns=list(fmargecol.index),
            index=list(fmargecol.columns),
        )
        - 1
    )

    # Weights and svd of Tc
    weightedTc = _rmultiplication(
        _rmultiplication(Tc.T, np.sqrt(marge_col)).T, np.sqrt(marge_row)
    )
    U, s, V = SVD(weightedTc.T, svd_flip=False)
    ncp0 = min(len(weightedTc.iloc[0]), len(weightedTc), model.nf)
    U = U[:, :ncp0]
    V = V.T[:, :ncp0]
    s = s[:ncp0]
    tmp = V
    V = U
    U = tmp
    mult = np.sign(np.sum(V, axis=0))

    # final V
    mult1 = pd.DataFrame(
        np.array(pd.DataFrame(np.array(_rmultiplication(pd.DataFrame(V.T), mult)))).T
    )
    V = pd.DataFrame()
    for i in range(len(mult1)):
        V[i] = mult1.iloc[i] / np.sqrt(marge_col[i])
    V = np.array(V).T

    # final U
    mult1 = pd.DataFrame(
        np.array(pd.DataFrame(np.array(_rmultiplication(pd.DataFrame(U.T), mult)))).T
    )
    U = pd.DataFrame()
    for i in range(len(mult1)):
        U[i] = mult1.iloc[i] / np.sqrt(marge_row[i])
    U = np.array(U).T

    # computing the contribution
    eig: Any = s**2

    for i in range(len(V[0])):
        V[:, i] = V[:, i] * np.sqrt(eig[i])
    coord_col = V

    for i in range(len(U[0])):
        U[:, i] = U[:, i] * np.sqrt(eig[i])

    coord_col = coord_col**2

    for i in range(len(coord_col[0])):
        coord_col[:, i] = (coord_col[:, i] * marge_col) / eig[i]

    return coord_col * 100


def stats(model: Model, df: pd.DataFrame) -> Model:
    """Compute the contributions.

    Parameters:
        model: Model computed by fit.
        df : dataframe to compute contributions from in the original space

    Returns:
        model.
    """
    contributions = get_variable_contributions(model, df)
    model.contributions = contributions
    return model


def _rmultiplication(F: pd.DataFrame, marge: NDArray[Any]) -> pd.DataFrame:
    """Multiply each column with the same vector."""
    df_dict = F.to_dict("list")
    for col in df_dict.keys():
        df_dict[col] = df_dict[col] * marge
    df = pd.DataFrame.from_dict(df_dict)
    df.index = F.index
    return df


def _rdivision(F: pd.DataFrame, marge: NDArray[Any]) -> pd.DataFrame:
    """Divide each column with the same vector."""
    df_dict = F.to_dict("list")
    for col in df_dict.keys():
        df_dict[col] = df_dict[col] / marge
    df = pd.DataFrame.from_dict(df_dict)
    df.index = F.index
    return df
