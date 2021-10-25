"""FAMD projection."""
import sys
from itertools import chain, repeat
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

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
    col_w: Optional[ArrayLike] = None,
    scale: Optional[bool] = True,
) -> Tuple[pd.DataFrame, Model, Parameters]:
    """Project data into a lower dimensional space using FAMD.

    Args:
        df: data to project
        nf: number of components to keep (default: {min(df.shape[0], 5)})
        col_w: importance of each variable in the projection
            (more weight = more importance in the axes)
        scale: not used

    Returns:
        The transformed variables, model and parameters.
    """
    nf = nf or min(df.shape)
    col_w = col_w or np.ones(df.shape[1])
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    fit_check_params(nf, col_w, df.shape[1])

    # select the categorical and continuous columns
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.values
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.values

    row_w = row_weights_uniform(len(df))
    col_w = col_weights_compute(df, col_w, quanti, quali)

    df_scale, mean, std, prop, _modalities = center(df, quanti, quali)

    # apply the weights
    Z = ((df_scale * col_w).T * row_w).T

    # compute the svd
    U, s, V = SVD(Z)
    U = ((U.T) / np.sqrt(row_w)).T
    V = V / np.sqrt(col_w)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    columns = column_names(nf)

    coord = pd.DataFrame(np.dot(df_scale, V.T), columns=columns)

    model = Model(
        df=df,
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
    )

    param = Parameters(
        nf=nf, col_w=col_w, row_w=row_w, columns=columns, quanti=quanti, quali=quali
    )

    return coord, model, param


def col_weights_compute(
    df: pd.DataFrame, col_w: list, quanti: list, quali: list
) -> Tuple[list, list]:
    """Initiate the weight vectors."""
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

    col_w = list(weight_quanti.iloc[0]) + weight_quali_rep

    return col_w


def center(
    df: pd.DataFrame, quanti: list, quali: list
) -> Tuple[pd.DataFrame, float, float, float, list]:
    """Scale data and compute mean, pro and std."""
    # Scale the continuous data
    df_quanti = df[quanti]
    mean = np.mean(df_quanti, axis=0)
    df_quanti -= mean
    std = np.std(df_quanti, axis=0)
    std[std <= sys.float_info.min] = 1
    df_quanti /= std

    # scale the categorical data
    df_quali = pd.get_dummies(df[quali].astype("category"))
    prop = np.mean(df_quali, axis=0)
    df_quali -= prop
    df_quali /= np.sqrt(prop)
    _modalities = df_quali.columns.values

    df_scale = pd.concat([df_quanti, df_quali], axis=1)

    return df_scale, mean, std, prop, _modalities


def transform(df: pd.DataFrame, model: Model, param: Parameters) -> pd.DataFrame:
    """Scale and project into the fitted numerical space."""
    df_scaled = scaler(model, param, df)
    return pd.DataFrame(np.dot(df_scaled, model.V.T), columns=param.columns)


def scaler(
    model: Model, param: Parameters, df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Scale data using prop, std and mean."""
    if df is None:
        df = model.df

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df_quanti = df[param.quanti]
    df_quanti = (df_quanti - model.mean) / model.std

    # scale
    df_quali = pd.get_dummies(df[param.quali].astype("category"))
    for mod in model._modalities:
        if mod not in df_quali:
            df_quali[mod] = 0
    df_quali = df_quali[model._modalities]
    df_quali = (df_quali - model.prop) / np.sqrt(model.prop)

    df_scale = pd.concat([df_quanti, df_quali], axis=1)
    return df_scale


def stats(model: Model, param: Parameters) -> Parameters:
    """Compute contributions and cos2 for each variable."""
    df = pd.DataFrame(scaler(model, param))
    df2 = np.array(pd.DataFrame(df).applymap(lambda x: x ** 2))

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
    eig = s ** 2
    # end of the svd

    # compute the contribution
    coord_var = np.array(V[0] * s)
    for i in range(1, len(V[:, 0])):
        coord_var = np.vstack((coord_var, V[i] * s))
    contrib_var = (((((coord_var ** 2) / eig).T) * param.col_w).T) * 100
    # compute cos2
    dfrow_w = ((df2.T) * param.row_w).T
    dist2 = []
    for i in range(len(dfrow_w[0])):
        dist2 += [np.sum(dfrow_w[:, i])]
        if abs(abs(dist2[i]) - 1) < 0.001:
            dist2[i] = 1

    cor = ((coord_var.T) / np.sqrt(dist2)).T
    cos2 = cor ** 2

    # compute eta2
    model.df.index = range(len(model.df))
    dfquali = model.df[param.quali]
    eta2 = []
    fi = 0
    coord = pd.DataFrame(
        model.U[:, :ncp0] * model.s[:ncp0], columns=param.columns[:ncp0]
    )
    mods = []
    # for each qualitative column in the original data set
    for count, col in enumerate(dfquali.columns):
        dummy = pd.get_dummies(dfquali[col].astype("category"))
        mods += [len(dummy.columns) - 1]
        # for each dimension
        dim = []
        for j, coordcol in enumerate(coord.columns):
            # for each modality of the qualitative column
            s = 0
            for i in range(len(dummy.columns)):
                s += (
                    np.array(dummy.T)[i] * coord[coordcol] * param.row_w
                ).sum() ** 2 / model.prop[fi + i]
            dim += [s]
        eta1 = (
            np.array(dim) / (np.array((coord ** 2)).T * param.row_w).sum(axis=1)
        ).tolist()
        eta2 += [eta1]
        fi += len(dummy.columns)

        cos2 = cos2[: len(param.quanti)]

    cos2 = cos2 ** 2
    eta2 = np.array(eta2) ** 2
    eta2 = (eta2.T / mods).T

    cos2 = np.concatenate([cos2, eta2], axis=0)
    param.contrib = contrib_var
    param.cos2 = cos2
    return param


def _rmultiplication(F: pd.DataFrame, marge: ArrayLike) -> pd.DataFrame:
    """Multiply each column with the same vector."""
    df_dict = F.to_dict("list")
    for col in df_dict.keys():
        df_dict[col] = df_dict[col] * marge
    df = pd.DataFrame.from_dict(df_dict)
    df.index = F.index
    return df
