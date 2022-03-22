"""Project any dataframe, inverse transform and compute stats."""
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import saiph.reduction.famd as famd
import saiph.reduction.mca as mca
import saiph.reduction.pca as pca
from saiph.models import Model, Parameters
from saiph.reduction import DUMMIES_PREFIX_SEP


def fit(
    df: pd.DataFrame,
    nf: Optional[Union[int, str]] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model, Parameters]:
    """Fit a PCA, MCA or FAMD model on data, imputing what has to be used.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters
    ----------
    df: pd.DataFrame
        Data to project.
    nf: int|str, default: 'all'
        Number of components to keep.
    col_w: np.ndarrayn default: np.ones(df.shape[1])
        Weight assigned to each variable in the projection
        (more weight = more importance in the axes).

    Returns
    -------
    coord: pd.DataFrame
        The transformed data.
    model: Model
        The model for transforming new data.
    param: Parameters
        The parameters for transforming new data.
    """
    # Check column types
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.values
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.values

    _nf: int
    if not nf or isinstance(nf, str):
        _nf = min(pd.get_dummies(df, prefix_sep=DUMMIES_PREFIX_SEP).shape)
    else:
        _nf = nf

    # Specify the correct function
    if quali.size == 0:
        _fit = pca.fit
    elif quanti.size == 0:
        _fit = mca.fit
    else:
        _fit = famd.fit

    coord, model, param = _fit(df, _nf, col_w)

    param.correlations = _variable_correlation(model, df, param)

    if quanti.size == 0:
        model.variable_coord = pd.DataFrame(model.D_c @ model.V.T)
    else:
        model.variable_coord = pd.DataFrame(model.V.T)

    return coord, model, param


def stats(model: Model, param: Parameters, df: pd.DataFrame) -> Parameters:
    """Compute the contributions and cos2.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    param: Parameters
        Param computed by fit.
    df : pd.DataFrame
        original dataframe

    Returns
    -------
    param: Parameters
        param populated with contriubtion.
    """
    # Check attributes type
    if not model.is_fitted:
        raise ValueError(
            "empty param, run fit function to create Model class and Parameters class objects"
        )

    model.variable_coord.columns = param.correlations.columns
    model.variable_coord.index = list(param.correlations.index)

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    if not has_some_quali:
        param.cos2 = param.correlations**2
        param.contributions = param.cos2.div(param.cos2.sum(axis=0), axis=1).mul(100)
    elif not has_some_quanti:
        param = mca.stats(model, param, df)
        param.cos2 = param.correlations**2

        param.contributions = pd.DataFrame(
            param.contributions,
            columns=param.correlations.columns,
            index=list(param.correlations.index),
        )
    else:
        param = famd.stats(model, param, df)
        param.cos2 = pd.DataFrame(
            param.cos2, index=model.original_continuous + model.original_categorical
        )
        param.contributions = pd.DataFrame(
            param.contributions,
            columns=param.correlations.columns,
            index=list(param.correlations.index),
        )
    return param


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
    if not model.is_fitted:
        raise ValueError("Need to fit before using transform")

    if len(model.original_categorical) == 0:
        coord = pca.transform(df, model, param)
    elif len(model.original_continuous) == 0:
        coord = mca.transform(df, model, param)
    else:
        coord = famd.transform(df, model, param)
    return coord


def _variable_correlation(
    model: Model,
    df: pd.DataFrame,
    param: Parameters,
) -> pd.DataFrame:
    """Compute the correlation between the axis and the variables.

    Args:
        model (Model): the model
        df (pd.DataFrame): dataframe
        quali : categorical variables
        quali : continuous variables

    Returns:
        pd.DataFrame: correlations between the axis and the variables
    """
    # select columns and project data
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )
    df_quanti = df[model.original_continuous]
    coord = transform(df, model, param)  # transform to be
    if has_some_quali:
        df_quali = pd.get_dummies(
            df[model.original_categorical].astype("category"),
            prefix_sep=DUMMIES_PREFIX_SEP,
        )
        bind = pd.concat([df_quanti, df_quali], axis=1)
    else:
        bind = df_quanti

    concat = pd.concat([bind, coord], axis=1, keys=["bind", "coord"])
    cor = pd.DataFrame(np.corrcoef(concat, rowvar=False), columns=concat.columns).loc[
        0 : len(bind.columns) - 1, "coord"
    ]
    return cor


def inverse_transform(
    coord: pd.DataFrame,
    model: Model,
    param: Parameters,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute the inverse transform of data coordinates.

    Note that if nf was stricly smaller than max(df.shape) in fit,
    inverse_transform o transform != id

    Parameters
    ----------
    coord: pd.DataFrame
        DataFrame to transform.
    model: Model
        Model computed by fit.
    param: Parameters
        Param computed by fit.
    seed: int|None, default: None
        Specify the seed for np.random.

    Returns
    -------
    inverse: pd.DataFrame
        Inversed DataFrame.
    """
    if len(coord) < param.nf:
        raise ValueError(
            "Inverse_transform is not working"
            "if the number of dimensions is greater than the number of individuals"
        )

    inverse = inverse_transform_raw(coord, model, param, seed)

    # Cast columns to same type as input
    for name, dtype in model.original_columns.iteritems():
        inverse[name] = inverse[name].astype(dtype)

    # reorder columns
    return inverse[model.original_columns.index]


def inverse_transform_raw(
    coord: pd.DataFrame, model: Model, param: Parameters, seed: Optional[int] = None
) -> pd.DataFrame:

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    inverse_coord_quanti, inverse_coord_quali = inverse_coordinates(coord, model, param)

    if not has_some_quanti:
        inverse_coord_quali = coord @ (model.D_c @ model.V.T).T
        inverse_coord_quali = np.divide(inverse_coord_quali, param.dummies_col_prop)

        # dividing by proportion of each modality among individual
        # allows to get back the complete disjunctive table
        # X_quali is the complete disjunctive table ("tableau disjonctif complet" in FR)

    inverse_quanti = (
        inverse_transform_quanti(inverse_coord_quanti, model)
        if has_some_quanti
        else None
    )

    # Rescale quali
    # FIXME: Can we avoid doing an operation on just X_quali when we have both quanti and quali?
    if has_some_quanti and has_some_quali:
        prop = model.prop.to_numpy()
        inverse_coord_quali = (inverse_coord_quali * np.sqrt(prop)) + prop

    inverse_quali = (
        inverse_transform_quali(pd.DataFrame(inverse_coord_quali), model, seed)
        if has_some_quali
        else None
    )

    if inverse_quali is None:
        return inverse_quanti
    elif inverse_quanti is None:
        return inverse_quali
    else:
        return pd.concat([inverse_quali, inverse_quanti], axis=1)


def inverse_coordinates(
    coord: NDArray[np.float_], model: Model, param: Parameters
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    # Inverse
    inverse_coords: NDArray[np.float_] = np.array(
        coord @ model.V * np.sqrt(param.column_weights)
    )

    # Scale
    inverse_coords = (
        inverse_coords / np.sqrt(param.column_weights) * param.column_weights
    )

    nb_quanti = len(model.original_continuous)

    inverse_coord_quanti = inverse_coords[:, :nb_quanti]
    inverse_coord_quali = inverse_coords[:, nb_quanti:]

    return inverse_coord_quanti, inverse_coord_quali


def inverse_transform_quanti(
    inverse_coords: NDArray[np.float_],
    model: Model,
) -> pd.DataFrame:
    std: NDArray[np.float_] = model.std.to_numpy()
    mean: NDArray[np.float_] = model.mean.to_numpy()
    inverse_quanti = pd.DataFrame(
        data=(inverse_coords * std) + mean,
        columns=model.original_continuous,
    )
    # FIXME: Why are we rounding here ? Removing it makes tests fail.
    return inverse_quanti.round(1)


def inverse_transform_quali(
    inverse_coords: pd.DataFrame,
    model: Model,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Inverse transform categorical variables by weighted random selection.

    Args:
        train_df (pd.DataFrame): dataframe used to fit the model
        seed (int): seed to fix randomness
        inverse_coords (pd.DataFrame): coordinates that we want to transform

    Returns:
        inverse_quali: transformed categorical dataframe that is similar to `train_df`

    """
    inverse_coords.columns = model.dummy_categorical

    dummies_mapping = get_dummies_mapping(
        model.original_categorical, model.dummy_categorical
    )

    random_gen = np.random.default_rng(seed)
    inverse_quali = pd.DataFrame()

    def get_suffix(string: str) -> str:
        return string.split(DUMMIES_PREFIX_SEP)[1]

    for original_column, dummy_columns in dummies_mapping.items():
        # Handle a single category with all the possible modalities
        single_category = inverse_coords[dummy_columns]
        chosen_modalities = get_random_weighted_columns(single_category, random_gen)
        inverse_quali[original_column] = list(map(get_suffix, chosen_modalities))

    return inverse_quali


def get_random_weighted_columns(
    df: pd.DataFrame, random_gen: np.random.Generator
) -> pd.Series:
    """Randomly select column labels weighted by proportions.

    Args:
        df : dataframe containing proportions
        random_gen (np.random.Generator): random generator

    Returns:
        selected column labels
    """
    # Example for 1 row:  [0.1, 0.3, 0.6] --> [0.1, 0.4, 1.0]
    cum_probability = df.cumsum(axis=1)
    random_probability = random_gen.random((cum_probability.shape[0], 1))
    # [0.342] < [0.1, 0.4, 1.0] --> [False, True, True] --> idx: 1
    column_labels = (random_probability < cum_probability).idxmax(axis=1)

    return column_labels


def get_dummies_mapping(
    columns: List[str], dummy_columns: List[str]
) -> Dict[str, Set[str]]:
    """Get mapping between original column and all dummy columns."""
    return {
        col: set(
            filter(lambda c: c.startswith(f"{col}{DUMMIES_PREFIX_SEP}"), dummy_columns)
        )
        for col in columns
    }
