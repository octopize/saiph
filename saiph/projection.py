"""Project any dataframe, inverse transform and compute stats."""
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import saiph.reduction.famd as famd
import saiph.reduction.mca as mca
import saiph.reduction.pca as pca
from saiph.models import Model, Parameters


def fit(
    df: pd.DataFrame,
    *,
    nf: Optional[int] = None,
    col_w: Optional[NDArray[np.float_]] = None,
    scale: Optional[bool] = True,
    approximate : bool = False,
    algorithm : Optional[str] = 'randomized',
    n_components : Optional[int] = None

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
        _nf = min(pd.get_dummies(df).shape)
    else:
        _nf = nf

    # Specify the correct function
    if quali.size == 0:
        _fit = pca.fit
    elif quanti.size == 0:
        _fit = mca.fit
    else:
        _fit = famd.fit

    coord, model, param = _fit(df, nf=_nf, col_w=col_w, scale=scale, algorithm=algorithm)
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
        param.cos2 = param.cor.applymap(lambda x: x ** 2)
        param.contrib = param.cos2.div(param.cos2.sum(axis=0), axis=1).applymap(
            lambda x: x * 100
        )
    elif param.quanti.size == 0:
        param = mca.stats(model, param)
        if param.cor is None:
            raise ValueError(
                "empty param, run fit function to create Model class and Parameters class objects"
            )
        param.cos2 = param.cor.applymap(lambda x: x ** 2)
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
        df_quali = pd.get_dummies(model.df[param.quali].astype("category"))
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

    # if PCA or FAMD compute the continuous variables
    if param.quanti is not None and len(param.quanti) != 0:

        X: NDArray[np.float_] = np.array(coord @ model.V * np.sqrt(param.col_w))
        X = X / np.sqrt(param.col_w) * param.col_w
        X_quanti = X[:, : len(param.quanti)]

        # descale
        std: NDArray[np.float_] = np.array(model.std)
        mean: NDArray[np.float_] = np.array(model.mean)
        inverse_quanti = (X_quanti * std) + mean
        # Handle floats -> int conversion. decimals=14 brings errors when casting as int
        inverse_quanti = pd.DataFrame(inverse_quanti, columns=list(param.quanti)).round(
            decimals=13
        )

        # if FAMD descale the categorical variables
        if param.quali is not None and len(param.quali) != 0:
            X_quali = X[:, len(param.quanti) :]
            prop: NDArray[np.float_] = np.array(model.prop)
            X_quali = (X_quali) * (np.sqrt(prop)) + prop

    # if MCA no descaling
    else:
        # Previously was a ndarray, but no need
        # NB: If this causes a bug, X_quali = np.array(X_quali) goes back to previous vesrion
        X_quali = coord @ (model.D_c @ model.V.T).T
        X_quali = np.divide(X_quali, param.dummies_col_prop)
        # divinding by proportion of each modality among individual
        # allows to get back the complete disjunctive table
        # X_quali is the complete disjunctive table ("tableau disjonctif complet" in FR)

    # compute the categorical variables
    if param.quali is not None and len(param.quali) != 0:
        inverse_quali = pd.DataFrame()
        X_quali = pd.DataFrame(X_quali)
        X_quali.columns = list(
            pd.get_dummies(
                model.df[param.quali],
                prefix=None,
                prefix_sep="_",
            ).columns
        )

        modalities = []
        for column in param.quali:
            modalities += [len(model.df[column].unique())]
        val = 0
        # conserve the modalities in their original type
        modalities_type = []
        for col in param.quali:
            mod_temp = list(model.df[col].unique())
            mod_temp.sort()  # sort the modalities as pd.get_dummies have done
            modalities_type += mod_temp

        # create a dict that link dummies variable to the original modalitie
        dict_mod = dict(zip(X_quali.columns, modalities_type))

        # for each variable we affect the value to the highest modalitie in X_quali
        random_gen = np.random.default_rng(seed)
        for i in range(len(modalities)):
            # get cumululative probabilities
            cum_probability = X_quali.iloc[:, val : val + modalities[i]].cumsum(axis=1)
            # random draw
            random_probability = random_gen.random((len(cum_probability), 1))
            # choose the modality according the probabilities of each modalities
            mod_random = (random_probability < cum_probability).idxmax(axis=1)
            mod_random = [dict_mod.get(x, x) for x in mod_random]
            inverse_quali[list(model.df[param.quali].columns)[i]] = mod_random
            val += modalities[i]

    # concatenate the continuous and categorical
    if (
        param.quali is not None
        and param.quanti is not None
        and len(param.quali) != 0
        and len(param.quanti) != 0
    ):
        inverse = pd.concat([inverse_quali, inverse_quanti], axis=1)
    elif param.quanti is not None and len(param.quanti) != 0:
        inverse = inverse_quanti
    else:
        inverse = inverse_quali

    # Cast columns to same type as input
    for column in model.df.columns:
        column_type = model.df.loc[:, column].dtype
        inverse[column] = inverse[column].astype(column_type)

    # reorder columns
    return inverse[model.df.columns]
