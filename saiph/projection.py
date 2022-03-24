"""Project any dataframe, inverse transform and compute stats."""
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_PREFIX_SEP, famd, mca, pca


def fit(
    df: pd.DataFrame,
    nf: Optional[Union[int, str]] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model]:
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

    coord, model = _fit(df, _nf, col_w)

    if quanti.size == 0:
        model.variable_coord = pd.DataFrame(model.D_c @ model.V.T)
    else:
        model.variable_coord = pd.DataFrame(model.V.T)

    return coord, model


def stats(model: Model, df: pd.DataFrame) -> Model:
    """Compute the contributions and cos2.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    df : pd.DataFrame
        original dataframe

    Returns
    -------
    model: Model
        model populated with contriubtion.
    """
    # Check attributes type
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    model.correlations = _variable_correlation(model, df)
    model.variable_coord.columns = model.correlations.columns
    model.variable_coord.index = list(model.correlations.index)

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    if not has_some_quali:
        model.cos2 = model.correlations**2
        model.contributions = model.cos2.div(model.cos2.sum(axis=0), axis=1).mul(100)
    elif not has_some_quanti:
        model = mca.stats(model, df)
        model.cos2 = model.correlations**2

        model.contributions = pd.DataFrame(
            model.contributions,
            columns=model.correlations.columns,
            index=list(model.correlations.index),
        )
    else:
        model = famd.stats(model, df)
        model.cos2 = pd.DataFrame(
            model.cos2, index=model.original_continuous + model.original_categorical
        )
        model.contributions = pd.DataFrame(
            model.contributions,
            columns=model.correlations.columns,
            index=list(model.correlations.index),
        )
    return model


def transform(df: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to transform.
    model: Model
        Model computed by fit.

    Returns
    -------
    coord: pd.DataFrame
        Coordinates of the dataframe in the fitted space.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted."
            "Call fit() to create a Model instance before calling transform()."
        )

    if len(model.original_categorical) == 0:
        coord = pca.transform(df, model)
    elif len(model.original_continuous) == 0:
        coord = mca.transform(df, model)
    else:
        coord = famd.transform(df, model)
    return coord


def _variable_correlation(
    model: Model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the correlation between the axis and the variables.

    Args:
        model (Model): the model
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: correlations between the axis and the variables
    """
    # select columns and project data
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )
    df_quanti = df[model.original_continuous]
    coord = transform(df, model)  # transform to be
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
    *,
    use_approximate_inverse: bool = False,
    use_max_modalities: bool = True,
) -> pd.DataFrame:
    """Return original format dataframe from coordinates.

    Arguments:
        coord (pd.DataFrame): coord of individuals to reverse transform
        model (Model): model used for projection
        use_approximate_inverse (boolean): matrix is not invertible when n_individuals < n_dimensions
            an approximation with bias can be done if necessary
        use_max_modalities (boolean): for each variable, it assign to the individual
            the value of the highest modality (True)
            or a random modality weighted by their proportion (False)

    Returns:
        pd.DataFrame: original shape, encoding and structure
    """
    # Check dimension size regarding N
    n_dimensions = len(model.dummy_categorical) + len(model.original_continuous)
    n_records = len(coord)

    if not use_approximate_inverse and n_records < n_dimensions:
        raise ValueError(
            f"n_dimensions ({n_dimensions}) is greater than n_records ({n_records})."           
            "A matrix approximation is needed but will intriduce bias "
            "You can reduce number of dimensions or set approximate=True."
        )

    # Get back scaled_values from coord with inverse matrix operation
    # If n_records < n_dimensions, There will be an approximation of the inverse of V.T
    scaled_values = pd.DataFrame(coord @ np.linalg.pinv(model.V.T))

    # get number of continuous variables
    nb_quanti = len(model.original_continuous)

    # separate quanti from quali
    scaled_values_quanti = scaled_values.iloc[:, :nb_quanti]
    scaled_values_quanti.columns = model.original_continuous

    scaled_values_quali = scaled_values.iloc[:, nb_quanti:]
    scaled_values_quali.columns = model.dummy_categorical

    # Descale regarding projection type
    # FAMD
    if model.type == "famd":
        descaled_values_quanti = (scaled_values_quanti * model.std) + model.mean
        descaled_values_quali = (scaled_values_quali * np.sqrt(model.prop)) + model.prop
        undummy = undummify(
            descaled_values_quali, model, use_max_modalities=use_max_modalities
        )
        inverse = pd.concat([descaled_values_quanti, undummy], axis=1).round(12)

    # PCA
    elif model.type == "pca":
        descaled_values_quanti = (scaled_values_quanti * model.std) + model.mean
        inverse = descaled_values_quanti.round(12)

    # MCA
    else:
        descaled_values_quali = scaled_values_quali * scaled_values_quali.sum().sum()
        inverse = undummify(
            descaled_values_quali, model, use_max_modalities=use_max_modalities
        )

    # Cast columns to same type as input
    for name, dtype in model.original_dtypes.iteritems():
        inverse[name] = inverse[name].astype(dtype)

    # reorder columns
    return inverse[model.original_dtypes.index]


def undummify(
    dummy_df: pd.DataFrame,
    model: Model,
    *,
    use_max_modalities: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return undummified dataframe from the dummy.

    Arguments:
        dummy_df (pd.DataFrame): dummy df of categorical variable
        model (Model): model used for projection
        use_max_modalities (boolean): True to set quali modality to max value
        seed (int): seed to fix randomness if use_max_modalities = False

    Returns:
        pd.DataFrame: undummify df of categorical variable
    """
    dummies_mapping = get_dummies_mapping(
        model.original_categorical, model.dummy_categorical
    )
    inverse_quali = pd.DataFrame()
    random_gen = np.random.default_rng(seed)
    def get_suffix(string: str) -> str:
        return string.split(DUMMIES_PREFIX_SEP)[1]

    for original_column, dummy_columns in dummies_mapping.items():
        # Handle a single category with all the possible modalities
        single_category = dummy_df[dummy_columns]

        # if use_max_modalities set max value per row to 1 and 0 for other
        if use_max_modalities:
            chosen_modalities = single_category.idxmax(axis='columns')
        else:
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
