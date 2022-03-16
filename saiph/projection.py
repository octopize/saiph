"""Project any dataframe, inverse transform and compute stats."""
from typing import Dict, List, Optional, Tuple, Union

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
    scale: bool = True,
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

    coord, model, param = _fit(df, _nf, col_w, scale)
    param.quanti = quanti
    param.quali = quali
    param.cor = _variable_correlation(model, param)

    if quanti.size == 0:
        model.variable_coord = pd.DataFrame(model.D_c @ model.V.T)
    else:
        model.variable_coord = pd.DataFrame(model.V.T)

    return coord, model, param


def stats(model: Model, param: Parameters) -> Parameters:
    """Compute the contributions and cos2.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    param: Parameters
        Param computed by fit.

    Returns
    -------
    param: Parameters
        param populated with contriubtion.
    """
    # Check attributes type
    if param.cor is None or param.quanti is None or param.quali is None:
        raise ValueError(
            "empty param, run fit function to create Model class and Parameters class objects"
        )
    model.variable_coord.columns = param.cor.columns
    model.variable_coord.index = list(param.cor.index)

    if param.quali.size == 0:
        param.cos2 = param.cor ** 2
        param.contrib = param.cos2.div(param.cos2.sum(axis=0), axis=1).mul(100)
    elif param.quanti.size == 0:
        param = mca.stats(model, param)
        if param.cor is None:
            raise ValueError(
                "empty param, run fit function to create Model class and Parameters class objects"
            )
        param.cos2 = param.cor ** 2

        param.contrib = pd.DataFrame(
            param.contrib,
            columns=param.cor.columns,
            index=list(param.cor.index),
        )
    else:
        param = famd.stats(model, param)
        if param.cor is None or param.quanti is None or param.quali is None:
            raise ValueError(
                "empty param, run fit function to create Model class and Parameters class objects"
            )
        param.cos2 = pd.DataFrame(
            param.cos2, index=list(param.quanti) + list(param.quali)
        )
        param.contrib = pd.DataFrame(
            param.contrib,
            columns=param.cor.columns,
            index=list(param.cor.index),
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
    if param.quali is None or param.quanti is None:
        raise ValueError("Need to fit before using transform")

    if param.quali.size == 0:
        coord = pca.transform(df, model, param)
    elif param.quanti.size == 0:
        coord = mca.transform(df, model, param)
    else:
        coord = famd.transform(df, model, param)
    return coord


def _variable_correlation(model: Model, param: Parameters) -> pd.DataFrame:
    """Compute the correlation between the axis and the variables."""
    # select columns and project data
    df_quanti = model.df[param.quanti]
    coord = transform(model.df, model, param)  # transform to be fixed

    if param.quali is not None and len(param.quali) > 0:
        df_quali = pd.get_dummies(
            model.df[param.quali].astype("category"), prefix_sep=DUMMIES_PREFIX_SEP
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

    inverse = inverse_transfrom_raw(coord, model, param, seed)

    # Cast columns to same type as input
    for column in model.df.columns:
        column_type = model.df.loc[:, column].dtype
        inverse[column] = inverse[column].astype(column_type)

    # reorder columns
    return inverse[model.df.columns]


def inverse_transfrom_raw(
    coord: pd.DataFrame, model: Model, param: Parameters, seed: Optional[int] = None
) -> pd.DataFrame:

    has_some_quanti = param.quanti is not None and param.quanti.shape[0] != 0
    has_some_quali = param.quali is not None and len(param.quali) != 0

    X: NDArray[np.float_] = np.array(coord @ model.V * np.sqrt(param.col_w))
    X = X / np.sqrt(param.col_w) * param.col_w
    nb_quanti = param.quanti.shape[0]

    X_quanti = X[:, :nb_quanti]
    X_quali = X[:, nb_quanti:]

    if not has_some_quanti:
        # We can just do a matrix multiplication and we're done.
        X_quali = coord @ (model.D_c @ model.V.T).T
        X_quali = np.divide(X_quali, param.dummies_col_prop)

        # dividing by proportion of each modality among individual
        # allows to get back the complete disjunctive table
        # X_quali is the complete disjunctive table ("tableau disjonctif complet" in FR)

    inverse_quanti = (
        inverse_transformm_quanti(model, param, X_quanti) if has_some_quanti else None
    )

    # Rescale quali
    if has_some_quanti and has_some_quali:
        prop = model.prop.to_numpy()
        X_quali = (X_quali * np.sqrt(prop)) + prop

    inverse_quali = (
        inverse_transform_quali(model.df[param.quali], seed, pd.DataFrame(X_quali))
        if has_some_quali
        else None
    )

    # concatenate the continuous and categorical
    if inverse_quali is None:
        return inverse_quanti
    elif inverse_quanti is None:
        return inverse_quali
    else:
        return pd.concat([inverse_quali, inverse_quanti], axis=1)


def inverse_transformm_quanti(
    model: Model,
    param: Parameters,
    X_quanti: NDArray[np.float_],
) -> pd.DataFrame:
    std: float = model.std.to_numpy()
    mean: float = model.mean.to_numpy()
    inverse_quanti = pd.DataFrame(
        data=(X_quanti * std) + mean,
        columns=param.quanti,
    )
    return inverse_quanti.round(1)


def inverse_transform_quali(
    train_df: pd.DataFrame,
    seed: int,
    X: pd.DataFrame,
) -> pd.DataFrame:

    dummy_columns = pd.get_dummies(train_df, prefix_sep=DUMMIES_PREFIX_SEP).columns
    X.columns = dummy_columns

    modalities = get_number_unique(train_df)
    dict_mod = get_column_mapping(dummy_columns)

    random_gen = np.random.default_rng(seed)
    inverse_quali = pd.DataFrame()

    index = 0
    for i, modality in enumerate(modalities):
        # get cumululative probabilities
        upper_index = index + modality
        cum_probability = X.iloc[:, index:upper_index].cumsum(axis=1)
        # random draw
        random_probability = random_gen.random((len(cum_probability), 1))
        # choose the modality according the probabilities of each modalities
        chosen_modalities = (random_probability < cum_probability).idxmax(axis=1)
        chosen_modalities = [dict_mod.get(x, x) for x in chosen_modalities]
        inverse_quali[train_df.columns[i]] = chosen_modalities
        index += modality
    return inverse_quali


def get_column_mapping(dummy_columns: List[str]) -> Dict[str, str]:
    """Get mapping between dummy columns and original columns."""
    return {col: col.split(DUMMIES_PREFIX_SEP)[1] for col in dummy_columns}


def get_number_unique(df: pd.DataFrame) -> List[int]:
    return df.nunique().to_list()  # type: ignore
