"""FAMD projection module."""
import sys
from itertools import chain, repeat
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model, Parameters
from saiph.reduction import DUMMIES_PREFIX_SEP
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
    col_w: Optional[NDArray[np.float_]] = None,
    scale: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Model, Parameters]:
    """Fit a FAMD model on data.

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
    if col_w is not None:
        _col_weights = col_w
    else:
        _col_weights = np.ones(df.shape[1])

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    fit_check_params(nf, _col_weights, df.shape[1])

    # select the categorical and continuous columns
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.to_list()
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.to_list()
    dummy_categorical = pd.get_dummies(
        df[quali].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    ).columns.to_list()

    row_w = row_weights_uniform(len(df))
    col_weights = _col_weights_compute(df, _col_weights, quanti, quali)

    df_scale, mean, std, prop, _modalities = center(df, quanti, quali)

    # apply the weights
    Z = ((df_scale * col_weights).T * row_w).T

    # compute the svd
    _U, s, _V = SVD(Z)
    U = ((_U.T) / np.sqrt(row_w)).T
    V = _V / np.sqrt(col_weights)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]
    columns = column_names(nf)
    coord = df_scale @ V.T
    coord.columns = columns

    model = Model(
        original_columns=df.dtypes,
        original_categorical=quali,
        original_continuous=quanti,
        dummy_categorical=dummy_categorical,
        U=U,
        V=V,
        s=s,
        explained_var=explained_var,
        explained_var_ratio=explained_var_ratio,
        variable_coord=pd.DataFrame(V.T),
        mean=mean,
        std=std,
        prop=prop,
        _modalities=_modalities,
        type="famd",
        is_fitted=True,
    )

    param = Parameters(
        nf=nf,
        col_w=col_weights,
        row_w=row_w,
        columns=columns,
    )

    return coord, model, param


def _col_weights_compute(
    df: pd.DataFrame, col_w: NDArray[Any], quanti: List[int], quali: List[int]
) -> NDArray[Any]:
    """Calculate weights for columns given what weights the user gave."""
    # Set the columns and row weights
    weight_df = pd.DataFrame([col_w], columns=df.columns)
    weight_quanti = weight_df[quanti]
    weight_quali = weight_df[quali]

    # Get the number of modality for each quali variable
    modality_numbers = []
    for column in weight_quali.columns:
        modality_numbers += [len(df[column].unique())]

    # Set weight vector for categorical columns
    weight_quali_rep = list(
        chain.from_iterable(
            repeat(i, j) for i, j in zip(list(weight_quali.iloc[0]), modality_numbers)
        )
    )

    _col_w: NDArray[Any] = np.array(list(weight_quanti.iloc[0]) + weight_quali_rep)

    return _col_w


def center(
    df: pd.DataFrame, quanti: List[int], quali: List[int]
) -> Tuple[
    pd.DataFrame, NDArray[np.float_], NDArray[np.float_], NDArray[Any], NDArray[Any]
]:
    """Center data, scale it, compute modalities and proportions of each categorical.

    Used as internal function during fit.

    **NB**: saiph.reduction.famd.scaler is better suited when a Model is already fitted.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to center.
    quanti: np.ndarray
        Indexes of continous variables.
    quali: np.ndarray
        Indexes of categorical variables.

    Returns
    -------
    df_scale: pd.DataFrame
        The scaled DataFrame.
    mean: pd.Series
        Mean of the input dataframe.
    std: pd.Series
        Standard deviation of the input dataframe. Returns nan as std if no std was asked.
    prop: pd.Series
        Proportion of each categorical.
    _modalities: np.ndarray
        Modalities for the MCA.
    """
    # Scale the continuous data
    df_quanti = df[quanti]
    mean = np.mean(df_quanti, axis=0)
    df_quanti -= mean
    std = np.std(df_quanti, axis=0)
    std[std <= sys.float_info.min] = 1
    df_quanti /= std

    # scale the categorical data
    df_quali = pd.get_dummies(
        df[quali].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    )
    prop = np.mean(df_quali, axis=0)
    df_quali -= prop
    df_quali /= np.sqrt(prop)
    _modalities = df_quali.columns.values

    df_scale = pd.concat([df_quanti, df_quali], axis=1)

    return df_scale, mean, std, prop, _modalities


def scaler(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """Scale data using mean, std, modalities and proportions of each categorical from model.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    df: pd.DataFrame
        DataFrame to scale.

    Returns
    -------
    df_scaled: pd.DataFrame
        The scaled DataFrame.
    """
    df_quanti = df[model.original_continuous]
    df_quanti = (df_quanti - model.mean) / model.std

    # scale
    df_quali = pd.get_dummies(
        df[model.original_categorical].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    )
    if model._modalities is not None:
        for mod in model._modalities:
            if mod not in df_quali:
                df_quali[mod] = 0
    df_quali = df_quali[model._modalities]
    df_quali = (df_quali - model.prop) / np.sqrt(model.prop)

    df_scaled = pd.concat([df_quanti, df_quali], axis=1)
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


def stats(model: Model, param: Parameters, original_df: pd.DataFrame) -> Parameters:
    """Compute contributions and cos2.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    param: Parameters
        Param computed by fit.

    Returns
    -------
    param: Parameters
        param populated with contriubtion ans cos2.
    """
    if not model.is_fitted:
        raise ValueError(
            "empty param, run fit function to create Model class and Parameters class objects"
        )

    df = pd.DataFrame(scaler(model, original_df))
    df2: NDArray[np.float_] = np.array(df) ** 2

    # svd of x with row_w and col_w
    weightedTc = _rmultiplication(
        _rmultiplication(df.T, np.sqrt(param.col_w)).T, np.sqrt(param.row_w)
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
        V[i] = mult1.iloc[i] / np.sqrt(param.col_w[i])
    V = np.array(V).T
    # final U
    mult1 = pd.DataFrame(
        np.array(pd.DataFrame(np.array(_rmultiplication(pd.DataFrame(U.T), mult)))).T
    )
    U = pd.DataFrame()
    for i in range(len(mult1)):
        U[i] = mult1.iloc[i] / np.sqrt(param.row_w[i])
    U = np.array(U).T
    eig: Any = s**2
    # end of the svd

    # compute the contribution
    coord_var: NDArray[np.float_] = np.array(V[0] * s)
    for i in range(1, len(V[:, 0])):
        coord_var = np.vstack((coord_var, V[i] * s))
    contrib_var = (((((coord_var**2) / eig).T) * param.col_w).T) * 100
    # compute cos2
    dfrow_w: NDArray[np.float_] = np.array(pd.DataFrame((df2.T) * param.row_w).T)
    dist2 = []
    for i in range(len(dfrow_w[0])):
        dist2 += [np.sum(dfrow_w[:, i])]
        if abs(abs(dist2[i]) - 1) < 0.001:
            dist2[i] = 1

    cor = ((coord_var.T) / np.sqrt(dist2)).T
    cos2 = cor**2

    # compute eta2
    eta2: NDArray[np.float_] = np.array([])
    fi = 0
    coord = pd.DataFrame(
        model.U[:, :ncp0] * model.s[:ncp0], columns=param.columns[:ncp0]
    )
    mods = []
    # for each qualitative column in the original data set
    for col in model.original_categorical:
        dummy = pd.get_dummies(
            original_df[col].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
        )
        mods += [len(dummy.columns) - 1]
        # for each dimension
        dim = []
        for j, coordcol in enumerate(coord.columns):
            # for each modality of the qualitative column
            p = 0
            for i in range(len(dummy.columns)):
                p += (
                    np.array(dummy.T)[i] * coord[coordcol] * param.row_w
                ).sum() ** 2 / model.prop[fi + i]
            dim += [p]
        eta1 = (
            np.array(dim) / np.array((coord**2).T * param.row_w).sum(axis=1).tolist()
        )
        eta2 = np.append(eta2, eta1)
        fi += len(dummy.columns)

        cos2 = cos2[: len(model.original_continuous)]

    cos2 = cos2**2
    eta2 = eta2**2
    eta2 = ((eta2).T / mods).T

    cos2 = np.concatenate([cos2, [eta2]], axis=0)
    param.contrib = contrib_var
    param.cos2 = cos2
    return param


def _rmultiplication(F: pd.DataFrame, marge: NDArray[Any]) -> pd.DataFrame:
    """Multiply each column with the same vector."""
    df_dict = F.to_dict("list")
    for col in df_dict.keys():
        df_dict[col] = df_dict[col] * marge
    df = pd.DataFrame.from_dict(df_dict)
    df.index = F.index
    return df
