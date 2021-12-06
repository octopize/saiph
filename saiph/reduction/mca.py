"""MCA projection."""
from itertools import chain, repeat
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model, Parameters
from saiph.reduction.utils.check_params import fit_check_params

# from scipy.sparse import diags
from saiph.reduction.utils.common import (
    column_names,
    diag,
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
    """Fit a MCA model on data.

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
        Unused. Kept for compatibility with model enabling scale=True|False.

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

    # initiate row and columns weights
    row_w = row_weights_uniform(len(df))

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
    df_dummies = pd.get_dummies(df.astype("category"))
    dummies_col_prop = len(df_dummies) / df_dummies.sum(axis=0)

    # apply the weights and compute the svd
    Z = ((T * col_weights).T * row_w).T
    U, s, V = SVD(Z)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    columns = column_names(nf)[: min(df_scale.shape)]
    coord = df_scale @ D_c @ V.T
    coord.columns = columns

    model = Model(
        df=df,
        U=U,
        V=V,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=D_c @ V.T,
        _modalities=_modalities,
        D_c=D_c,
        type="mca",
    )

    param = Parameters(
        nf=nf,
        col_w=col_weights,
        row_w=row_w,
        columns=columns,
        dummies_col_prop=dummies_col_prop,
    )

    return coord, model, param


def center(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Center data and compute modalities.

    Used as internal function during fit. Scaler is better suited when a Model is already fitted.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to center.

    Returns
    -------
    df_centered: pd.DataFrame
        The centered DataFrame.
    _modalities: np.ndarray
        Modalities for the MCA
    r: np.ndarray
        Sums line by line
    c: np.ndarray
        Sums column by column
    """
    df_scale = pd.get_dummies(df.astype("category"))
    _modalities = df_scale.columns.values

    # scale data
    df_scale /= df_scale.sum().sum()

    c = np.sum(df_scale, axis=0)
    r = np.sum(df_scale, axis=1)
    return df_scale, _modalities, r, c


def scaler(model: Model, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scale data using fitted model.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    df: pd.DataFrame
        DataFrame to scale.
        If nothing is specified, takes the DataFrame on which the model was fitted.

    Returns
    -------
    df_scaled: pd.DataFrame
        The scaled DataFrame.
    """
    if df is None:
        df = model.df

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df_scaled = pd.get_dummies(df.astype("category"))
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
    eps = np.finfo(float).eps
    if df_scale.shape[0] >= 10000:
        D_r = diag(1 / (eps + np.sqrt(r)), use_scipy=True)
    else:
        D_r = diag(1 / (eps + np.sqrt(r)), use_scipy=False)
    D_c = diag(1 / (eps + np.sqrt(c)), use_scipy=False)

    T = D_r @ (df_scale - np.outer(r, c)) @ D_c
    return df_scale / np.array(r)[:, None], T, D_c


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
    coord = df_scaled @ model.D_c @ model.V.T
    coord.columns = param.columns
    return coord


def stats(model: Model, param: Parameters) -> Parameters:
    """Compute the contributions of each variable in each axis."""
    V = np.dot(model.D_c, model.V.T)  # type: ignore
    total = pd.get_dummies(model.df.astype("category")).sum().sum()
    df = pd.get_dummies(model.df.astype("category"))
    F = df / total

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
    ncp0 = min(len(weightedTc.iloc[0]), len(weightedTc), param.nf)
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
    eig: Any = s ** 2

    for i in range(len(V[0])):
        V[:, i] = V[:, i] * np.sqrt(eig[i])
    coord_col = V

    for i in range(len(U[0])):
        U[:, i] = U[:, i] * np.sqrt(eig[i])

    coord_col = coord_col ** 2

    for i in range(len(coord_col[0])):
        coord_col[:, i] = (coord_col[:, i] * marge_col) / eig[i]

    param.contrib = coord_col * 100

    return param


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
