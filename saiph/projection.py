"""Project any dataframe, inverse transform and compute stats."""
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from saiph.models import Model
from saiph.reduction import DUMMIES_PREFIX_SEP, famd, famd_sparse, mca, pca
from saiph.reduction.utils.common import get_dummies_mapping, get_projected_column_names


def fit(
    df: pd.DataFrame,
    nf: Optional[Union[int, str]] = None,
    col_weights: Optional[NDArray[np.float_]] = None,
    sparse: bool = False,
) -> Model:
    """Fit a PCA, MCA or FAMD model on data, imputing what has to be used.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: 'all'
        col_w: Weight assigned to each variable in the projection
            (more weight = more importance in the axes).
            default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
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
    elif sparse:
        _fit = famd_sparse.fit
    else:
        _fit = famd.fit

    model = _fit(df, _nf, col_weights)

    if quanti.size == 0:
        model.variable_coord = pd.DataFrame(model.D_c @ model.V.T)
    else:
        model.variable_coord = pd.DataFrame(model.V.T)
    return model


def fit_transform(
    df: pd.DataFrame,
    nf: Optional[Union[int, str]] = None,
    col_w: Optional[NDArray[np.float_]] = None,
) -> Tuple[pd.DataFrame, Model]:
    """Fit a PCA, MCA or FAMD model on data, imputing what has to be used.

    Datetimes must be stored as numbers of seconds since epoch.

    Parameters:
        df: Data to project.
        nf: Number of components to keep. default: 'all'
        col_w: Weight assigned to each variable in the projection
            (more weight = more importance in the axes).
            default: np.ones(df.shape[1])

    Returns:
        model: The model for transforming new data.
    """
    model = fit(df, nf, col_w)
    coord = transform(df, model)
    return coord, model


def stats(model: Model, df: pd.DataFrame, explode: bool = False) -> Model:
    """Compute the contributions and cos2.

    Parameters:
        model: Model computed by fit.
        df: original dataframe
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False).
            Only valid for categorical variables.

    Returns:
        model: model populated with contribution.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    model.correlations = get_variable_correlation(model, df)
    model.variable_coord.columns = get_projected_column_names(
        model.variable_coord.shape[1]
    )
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
        model = mca.stats(model, df, explode=explode)
    else:
        model = famd.stats(model, df, explode=explode)

    return model


def get_variable_contributions(
    model: Model, df: pd.DataFrame, explode: bool = False
) -> pd.DataFrame:
    """Compute the contributions of the `df` variables within the fitted space.

    Parameters:
        model: Model computed by fit.
        df: dataframe to compute contributions from
        explode: whether to split the contributions of each modality (True)
            or sum them as the contribution of the whole variable (False)

    Returns:
        contributions
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted. Call fit() to create a Model instance."
        )

    has_some_quanti = (
        model.original_continuous is not None and len(model.original_continuous) != 0
    )
    has_some_quali = (
        model.original_categorical is not None and len(model.original_categorical) != 0
    )

    if not has_some_quali:
        correlations = get_variable_correlation(model, df)
        cos2 = correlations**2
        contributions = cos2.div(cos2.sum(axis=0), axis=1).mul(100)
        contributions = contributions.set_index(df.columns)
        return contributions

    if not has_some_quanti:
        return mca.get_variable_contributions(model, df, explode=explode)

    contributions, _ = famd.get_variable_contributions(model, df, explode=explode)
    return contributions


def transform(df: pd.DataFrame, model: Model, *, sparse: bool = False) -> pd.DataFrame:
    """Scale and project into the fitted numerical space.

    Parameters:
        df: DataFrame to transform.
        model: Model computed by fit.

    Returns:
        coord: Coordinates of the dataframe in the fitted space.
    """
    if not model.is_fitted:
        raise ValueError(
            "Model has not been fitted."
            "Call fit() to create a Model instance before calling transform()."
        )

    if len(model.original_categorical) == 0:
        return pca.transform(df, model)

    if len(model.original_continuous) == 0:
        return mca.transform(df, model)

    if sparse:
        return famd_sparse.transform(df, model)

    return famd.transform(df, model)


def get_variable_correlation(
    model: Model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the correlation between the axis and the variables.

    Parameters:
        model: the model
        df: dataframe

    Returns:
        cor: correlations between the axis and the variables
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
        0 : len(bind.columns) - 1,
        "coord",
    ]
    return cor


def inverse_transform(
    coord: pd.DataFrame,
    model: Model,
    *,
    use_approximate_inverse: bool = False,
    use_max_modalities: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Return original format dataframe from coordinates.

    Parameters:
        coord: coord of individuals to reverse transform
        model: model used for projection
        use_approximate_inverse: matrix is not invertible when n_individuals < n_dimensions
            an approximation with bias can be done by setting to ``True``. default: ``False``
        use_max_modalities: for each variable, it assigns to the individual
            the modality with the highest proportion (True)
            or a random modality weighted by their proportion (False). default: True
        seed: seed to fix randomness if use_max_modalities = False. default: None

    Returns:
        inverse: coordinates transformed into original space.
            Retains shape, encoding and structure.
    """
    # Check dimension size regarding N
    n_dimensions = len(model.dummy_categorical) + len(model.original_continuous)
    n_records = len(coord)

    if not use_approximate_inverse and n_records < n_dimensions:
        raise ValueError(
            f"n_dimensions ({n_dimensions}) is greater than n_records ({n_records})."
            "A matrix approximation is needed but will introduce bias "
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
            descaled_values_quali,
            model,
            use_max_modalities=use_max_modalities,
            seed=seed,
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
            descaled_values_quali,
            model,
            use_max_modalities=use_max_modalities,
            seed=seed,
        )
    # Cast columns to same type as input
    for name, dtype in model.original_dtypes.iteritems():
        # Can create a bug if a column is object but contains int and float values,
        # first, we force the value type of the first value of the original df
        if dtype in ["object", "category"]:
            if model.modalities_types[name] == "bool":
                inverse[name] = [eval(ele) for ele in inverse[name]]
            else:
                inverse[name] = inverse[name].astype(model.modalities_types[name])

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
    """Return undummified dataframe from the dummy dataframe.

    Parameters:
        dummy_df: dummy df of categorical variables
        model: model used for projection
        use_max_modalities: True to select the modality with the highest probability.
                            False for a weighted random selection. default: True
        seed: seed to fix randomness if use_max_modalities = False. default: None

    Returns:
        inverse_quali: undummify df of categorical variable
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

        if use_max_modalities:
            # select modalities with highest probability
            chosen_modalities = single_category.idxmax(axis="columns")
        else:
            chosen_modalities = get_random_weighted_columns(single_category, random_gen)
        inverse_quali[original_column] = list(map(get_suffix, chosen_modalities))

    return inverse_quali


def get_random_weighted_columns(
    df: pd.DataFrame, random_gen: np.random.Generator
) -> pd.Series:
    """Randomly select column labels weighted by proportions.

    Parameters:
        df : dataframe containing proportions
        random_gen: random generator

    Returns:
        column_labels: selected column labels
    """
    # Example for 1 row:  [0.1, 0.3, 0.6] --> [0.1, 0.4, 1.0]
    cum_probability = df.cumsum(axis=1)
    random_probability = random_gen.random((cum_probability.shape[0], 1))
    # [0.342] < [0.1, 0.4, 1.0] --> [False, True, True] --> idx: 1
    column_labels = (random_probability < cum_probability).idxmax(axis=1)

    return column_labels
