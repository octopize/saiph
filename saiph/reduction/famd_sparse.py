"""FAMD projection module."""
import sys
from itertools import chain, repeat
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from saiph.models import Model
from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.utils.check_params import fit_check_params
from saiph.reduction.utils.common import (
    explain_variance,
    get_modalities_types,
    get_projected_column_names,
    get_uniform_row_weights,
)


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
    print(nf)
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
    modalities_types = get_modalities_types(df[quali])

    row_w = get_uniform_row_weights(len(df))
    col_weights = _col_weights_compute(df, _col_weights, quanti, quali)

    df_scaled, mean, std, prop, _modalities = center(df, quanti, quali)
    print("df scale type", type(df_scaled))
    # apply the weights
    Z = df_scaled.multiply(col_weights).T.multiply(row_w).T

    # compute the svd
    _U, s, _V = SVD_sparse(Z)
    U = ((_U.T) / np.sqrt(row_w)).T
    V = _V / np.sqrt(col_weights)
    print("fit _V", _V.shape)
    print("fit V", V.shape)

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
        type="famd_sparse",
        is_fitted=True,
        nf=nf,
        column_weights=col_weights,
        row_weights=row_w,
        modalities_types=modalities_types,
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
    df: pd.DataFrame, quanti: List[str], quali: List[str]
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
    df_quanti = scipy.sparse.csr_matrix(df_quanti)

    # scale the categorical data
    df_quali = pd.get_dummies(
        df[quali].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    )
    _modalities = df_quali.columns
    df_quali = csr_matrix(df_quali)
    print("df quali type", type(df_quali))

    prop = np.mean(df_quali, axis=0).tolist()[0]
    df_quali /= np.sqrt(prop)
    df_scale = scipy.sparse.hstack([df_quanti, df_quali], format="csr")
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
    df_quanti = scipy.sparse.csr_matrix(df_quanti)

    # scale
    df_quali = pd.get_dummies(
        df[model.original_categorical].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
    )
    if model._modalities is not None:
        for mod in model._modalities:
            if mod not in df_quali:
                df_quali[mod] = 0
    df_quali = df_quali[model._modalities]
    df_quali = csr_matrix(df_quali)
    df_quali /= np.sqrt(model.prop)

    df_scaled = scipy.sparse.hstack([df_quanti, df_quali], format="csr")
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
    print("transform df scaled", df_scaled.shape)
    print("transform V", model.V.shape)

    coord = pd.DataFrame(df_scaled * model.V.T)
    print("coord shape", coord.shape)
    print("nf ", (model.nf))
    coord.columns = get_projected_column_names(model.nf - 1)
    return coord


def SVD_sparse(
    df: pd.DataFrame, svd_flip: bool = True
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute Singular Value Decomposition.

    Parameters:
        df: Matrix to decompose.
        svd_flip: Whether to use svd_flip on U and V or not.

    Returns:
        U: Unitary matrix having left singular vectors as columns.
        s: The singular values.
        V: Unitary matrix having right singular vectors as rows.
    """
    U, s, V = scipy.sparse.linalg.svds(df, k=(min(df.shape) - 1), solver="arpack")

    return U, s, V
