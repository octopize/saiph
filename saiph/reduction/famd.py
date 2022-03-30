"""FAMD projection module."""
import sys
from itertools import chain, repeat
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_PREFIX_SEP
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
    """Fit a FAMD model on data.

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

    # select the categorical and continuous columns
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.to_list()
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.to_list()
    dummy_categorical = pd.get_dummies(
        df[quali].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    ).columns.to_list()

    row_w = get_uniform_row_weights(len(df))
    col_weights = _col_weights_compute(df, _col_weights, quanti, quali)

    df_scaled, mean, std, prop, _modalities = center(df, quanti, quali)

    # apply the weights
    Z = ((df_scaled * col_weights).T * row_w).T

    # compute the svd
    _U, s, _V = SVD(Z)
    U = ((_U.T) / np.sqrt(row_w)).T
    V = _V / np.sqrt(col_weights)

    explained_var, explained_var_ratio = explain_variance(s, df, nf)

    U = U[:, :nf]
    s = s[:nf]
    V = V[:nf, :]

    model = Model(
        original_dtypes=df.dtypes,
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
        nf=nf,
        column_weights=col_weights,
        row_weights=row_w,
    )

    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[int] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a FAMD model on data and return transformed data.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: min(df.shape)
        col_w: Weight assigned to each variable in the projection
            (more weight = more importance in the axes). default: np.ones(df.shape[1])

    Returns:
        coord: The transformed data.
        model: The model for transforming new data.
    """
    model = fit(df, nf, col_w)
    coord = transform(df, model)
    return coord, model


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

    Parameters:
        df: DataFrame to center.
        quanti: Indices of continous variables.
        quali: Indices of categorical variables.

    Returns:
        df_scale: The scaled DataFrame.
        mean: Mean of the input dataframe.
        std: Standard deviation of the input dataframe.
        prop: Proportion of each categorical.
        _modalities: Modalities for the MCA.
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

    Parameters:
        model: Model computed by fit.
        df: DataFrame to scale.

    Returns:
        df_scaled: The scaled DataFrame.
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


def stats(model: Model, df: pd.DataFrame) -> Model:
    """Compute contributions and cos2.

    Parameters:
        model: Model computed by fit.

    Returns:
        model: model populated with contriubtion and cos2.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    contributions, cos2 = get_variable_contributions(model, df)
    model.contributions = contributions
    model.cos2 = cos2
    return model


def get_variable_contributions(
    model: Model, df: pd.DataFrame
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Compute the contributions of the `df` variables within the fitted space.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute contributions from

    Returns:
        tuple of contributions and cos2.
    """
    scaled_df = pd.DataFrame(scaler(model, df))
    df2: NDArray[np.float_] = np.array(scaled_df) ** 2

    # svd of x with row_w and col_w
    weightedTc = _rmultiplication(
        _rmultiplication(scaled_df.T, np.sqrt(model.column_weights)).T,
        np.sqrt(model.row_weights),
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
        V[i] = mult1.iloc[i] / np.sqrt(model.column_weights[i])
    V = np.array(V).T
    # final U
    mult1 = pd.DataFrame(
        np.array(pd.DataFrame(np.array(_rmultiplication(pd.DataFrame(U.T), mult)))).T
    )
    U = pd.DataFrame()
    for i in range(len(mult1)):
        U[i] = mult1.iloc[i] / np.sqrt(model.row_weights[i])
    U = np.array(U).T
    eig: Any = s**2
    # end of the svd

    # compute the contribution
    coord_var: NDArray[np.float_] = np.array(V[0] * s)
    for i in range(1, len(V[:, 0])):
        coord_var = np.vstack((coord_var, V[i] * s))
    contrib_var = (((((coord_var**2) / eig).T) * model.column_weights).T) * 100
    # compute cos2
    dfrow_w: NDArray[np.float_] = np.array(pd.DataFrame((df2.T) * model.row_weights).T)
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

    columns = get_projected_column_names(model.nf)[:ncp0]
    coord = pd.DataFrame(model.U[:, :ncp0] * model.s[:ncp0], columns=columns)
    mods = []
    # for each qualitative column in the original data set
    for col in model.original_categorical:
        dummy = pd.get_dummies(
            df[col].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
        )
        mods += [len(dummy.columns) - 1]
        # for each dimension
        dim = []
        for j, coordcol in enumerate(coord.columns):
            # for each modality of the qualitative column
            p = 0
            for i in range(len(dummy.columns)):
                p += (
                    np.array(dummy.T)[i] * coord[coordcol] * model.row_weights
                ).sum() ** 2 / model.prop[fi + i]
            dim += [p]
        eta1 = (
            np.array(dim)
            / np.array((coord**2).T * model.row_weights).sum(axis=1).tolist()
        )
        eta2 = np.append(eta2, eta1)
        fi += len(dummy.columns)

        cos2 = cos2[: len(model.original_continuous)]

    cos2 = cos2**2
    eta2 = eta2**2
    eta2 = ((eta2).T / mods).T

    cos2 = np.concatenate([cos2, [eta2]], axis=0)

    return contrib_var, cos2


def _rmultiplication(F: pd.DataFrame, marge: NDArray[Any]) -> pd.DataFrame:
    """Multiply each column with the same vector."""
    df_dict = F.to_dict("list")
    for col in df_dict.keys():
        df_dict[col] = df_dict[col] * marge
    df = pd.DataFrame.from_dict(df_dict)
    df.index = F.index
    return df
