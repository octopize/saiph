"""Project any dataframe, iverse transform and compute stats."""
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import saiph.reduction.famd as famd
import saiph.reduction.mca as mca
import saiph.reduction.pca as pca
from saiph.models import DFLike, Model, Parameters


def fit(
    df: pd.DataFrame,
    nf: Optional[Union[int, str]] = None,
    col_w: Optional[ArrayLike] = None,
    scale: bool = True,
) -> Tuple[DFLike, Model, Parameters]:
    """Project data into a lower dimensional space using PCA, MCA or FAMD.

    Args:
        df: data to project
        nf: number of components to keep (default: {min(df.shape[0], 5)})
        col_w: importance of each variable in the projection
            more weight = more importance in the axes)
        df: data to project
        nf: number of components to keep (default: {min(df.shape[0], 5)})
        col_w: importance of each variable in the projection
            (more weight = more importance in the axes)
        scale: whether to scale data or not (only for PCA)

    Returns:
        The transformed variables, model and parameters
    """
    datetime_variables = []
    for i in range(0, df.shape[1]):
        if df.iloc[:, i].dtype == ("datetime64[ns]"):
            df.iloc[:, i] = (
                df.iloc[:, i] - np.datetime64("1970-01-01T00:00:00Z")
            ) / np.timedelta64(1, "s")
            datetime_variables.append(i)

    # Check column types
    quanti = df.select_dtypes(include=["int", "float", "number"]).columns.values
    quali = df.select_dtypes(exclude=["int", "float", "number"]).columns.values

    if nf == "all":
        nf = len(pd.get_dummies(df).columns.values)
    if not nf:
        nf = min(5, len(pd.get_dummies(df).columns.values))

    # Specify the correct function
    if quali.size == 0:
        print("A PCA is performed for dimension reduction")
        fit = pca.fit
    elif quanti.size == 0:
        print("An MCA is performed for dimension reduction")
        fit = mca.fit
    else:
        print("FAMD is performed for dimension reduction")
        fit = famd.fit

    coord, model, param = fit(df, nf, col_w, scale)
    param.quanti = quanti
    param.quali = quali
    param.datetime_variables = datetime_variables
    param.cor = _variable_correlation(model, param)

    if param.quanti.size == 0:
        model.variable_coord = pd.DataFrame(np.dot(model.D_c, model.V.T))
    else:
        model.variable_coord = pd.DataFrame(model.V.T)

    return coord, model, param


def stats(model: Model, param: Parameters) -> Parameters:
    """Compute the correlation, contributions and cos2 for each variable."""
    model.variable_coord.columns = param.cor.columns
    model.variable_coord.index = list(param.cor.index)

    if param.quali.size == 0:
        param.cos2 = param.cor.applymap(lambda x: x ** 2)
        param.contrib = param.cos2.div(param.cos2.sum(axis=0), axis=1).applymap(
            lambda x: x * 100
        )
    elif param.quanti.size == 0:
        param = mca.stats(model, param)
        param.cos2 = param.cor.applymap(lambda x: x ** 2)
        param.contrib = pd.DataFrame(
            param.contrib,
            columns=param.cor.columns,
            index=list(param.cor.index),
        )
    else:
        param = famd.stats(model, param)
        param.cos2 = pd.DataFrame(
            param.cos2, index=list(param.quanti) + list(param.quali)
        )
        param.contrib = pd.DataFrame(
            param.contrib,
            columns=param.cor.columns,
            index=list(param.cor.index),
        )
    return param


def transform(df: DFLike, model: Model, param: Parameters) -> DFLike:
    """Project new data into the fitted numerical space."""
    for i in param.datetime_variables:
        df.iloc[:, i] = (
            df.iloc[:, i] - np.datetime64("1970-01-01T00:00:00Z")
        ) / np.timedelta64(1, "s")
    if param.quali.size == 0:
        coord = pca.transform(df, model, param)
    elif param.quanti.size == 0:
        coord = mca.transform(df, model, param)
    else:
        coord = famd.transform(df, model, param)
    return coord


def _variable_correlation(model: Model, param: Parameters) -> DFLike:
    """Compute the correlation between the axis' and the variables."""
    # select columns and project data
    df_quanti = model.df[param.quanti]
    coord = transform(model.df, model, param)  # transform to be fixed

    if len(param.quali) > 0:
        df_quali = pd.get_dummies(model.df[param.quali].astype("category"))
        bind = pd.concat([df_quanti, df_quali], axis=1)
    else:
        bind = df_quanti

    # compute correlations
    cor = pd.DataFrame(
        {
            component: {
                var: coord[component].corr(bind[var].reset_index(drop=True))
                for var in bind.columns
            }
            for component in coord.columns
        }
    )
    return cor


def inverse_transform(
    coord: DFLike, model: Model, param: Parameters, shuffle: bool = False
) -> DFLike:  # ---------------------------------------finish this
    """Compute the inverse transform of data coordinates."""
    # if PCA or FAMD compute the continuous variables
    if len(param.quanti) != 0:

        X = np.dot(coord, model.variable_coord.T)
        X_quanti = X[:, : len(param.quanti)]

        # descale
        std = np.array(model.std)
        mean = np.array(model.mean)
        inverse_quanti = (X_quanti * std) + mean
        inverse_quanti = pd.DataFrame(inverse_quanti, columns=list(param.quanti))

        # round to the right decimal
        for column in inverse_quanti.columns:
            inverse_quanti["decimals"] = model.df[column].apply(decimal_count)
            # shuffling the decimals for the avatarization
            if shuffle:
                inverse_quanti["decimals"] = np.random.permutation(
                    inverse_quanti["decimals"].values
                )

            inverse_quanti[column] = inverse_quanti[[column, "decimals"]].apply(
                lambda x: np.round(x[column], int(x["decimals"])), axis=1
            )
            inverse_quanti.drop(["decimals"], axis=1, inplace=True)

        # if FAMD descale the categorical variables
        if len(param.quali) != 0:
            X_quali = X[:, len(param.quanti) :]
            prop = np.array(model.prop)
            X_quali = (X_quali) * (np.sqrt(prop)) + prop

    # if MCA no descaling
    else:
        X_quali = np.dot(coord, np.dot(model.D_c, model.V.T).T)
        # X_quali is the full disjunctive table ("tableau disjonctif complet" in FR)

    # compute the categorical variables
    if len(param.quali) != 0:
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
        dict_mod = zip(X_quali.columns, modalities_type)
        dict_mod = dict(dict_mod)

        # for each variable we affect the value to the highest modalitie in X_quali
        for i in range(len(modalities)):
            mod_max = X_quali.iloc[:, val : val + modalities[i]].idxmax(axis=1)
            mod_max = [x if x not in dict_mod else dict_mod[x] for x in mod_max]
            inverse_quali[list(model.df[param.quali].columns)[i]] = mod_max
            val += modalities[i]

    # concatenate the continuous and categorical
    if len(param.quali) != 0 and len(param.quanti) != 0:
        inverse = pd.concat([inverse_quali, inverse_quanti], axis=1)
    elif len(param.quanti) != 0:
        inverse = inverse_quanti
    else:
        inverse = inverse_quali

    # Cast columns to same type as input
    for column in model.df.columns:
        column_type = model.df.loc[:, column].dtype
        inverse[column] = inverse[column].astype(column_type)

    # reorder back columns
    inverse = inverse[model.df.columns]

    # Turn back datetime variables to original dtype
    for i in param.datetime_variables:
        inverse.iloc[:, i] = (
            inverse.iloc[:, i] * np.timedelta64(1, "s")
        ) + np.datetime64("1970-01-01T00:00:00Z")

    return inverse


def decimal_count(number: int) -> int:
    """Compute number of decimals for each data point."""
    f = str(number)
    if "." in f:
        digits = f[::-1].find(".")
    else:
        digits = 0
    return digits
