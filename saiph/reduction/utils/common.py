from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from toolz import concat

from saiph.reduction import DUMMIES_SEPARATOR


def get_projected_column_names(n: int) -> list[str]:
    return [f"Dim. {i + 1}" for i in range(n)]


def get_uniform_row_weights(n: int) -> NDArray[np.float64]:
    return np.full(n, 1 / n, dtype=np.float64)


def row_multiplication(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Multiply each row of `df` with the corresponding value in `arr`."""
    return df.multiply(arr, axis="rows")


def column_multiplication(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Multiply each column of `df` with the corresponding value in `arr`."""
    return df.multiply(arr, axis="columns")


def row_division(df: pd.DataFrame, arr: NDArray[Any]) -> pd.DataFrame:
    """Divide each row of `df` with the corresponding value in `arr`."""
    return df.divide(arr, axis="rows")


def diag(arr: NDArray[Any], use_scipy: bool = False) -> NDArray[Any]:
    if use_scipy:
        return scipy.sparse.diags(arr)  # type: ignore
    else:
        return np.diag(arr)


def get_explained_variance(
    s: NDArray[np.float64],
    nb_individuals: int,
    nf: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    all_variance = (s**2) / (nb_individuals - 1)
    variance = all_variance[:nf]

    # We divide by the all_variance and not variance because
    # if nf < len(all_variance), we shouldn't sum up to 100%
    variance_sum = all_variance.sum()

    variance_ratio = (
        variance / variance_sum if variance_sum != 0 else np.full_like(variance, np.nan)
    )

    return variance, variance_ratio


def get_modalities_types(df: pd.DataFrame) -> dict[str, str]:
    modalities_types = {col: get_type_as_string(df.loc[0, col]) for col in df.columns}
    return modalities_types


def get_dummies_mapping(columns: list[str], dummy_columns: list[str]) -> dict[str, list[str]]:
    """Get mapping between original column and all dummy columns."""
    return OrderedDict(
        {
            col: list(filter(lambda c: c.startswith(f"{col}{DUMMIES_SEPARATOR}"), dummy_columns))
            for col in columns
        }
    )


def expand_column_weights(
    col_weights: NDArray[Any],
    columns: list[str],
    quanti: list[str],
    quali: list[str],
    dummy_categorical: list[str],
) -> NDArray[np.float64]:
    """Repeat each categorical column's weight once per dummy column it produced.

    The count must come from `dummy_categorical`, not from `nunique`: `pd.get_dummies`
    emits no indicator for a null, so distinct values overcount and the weight vector
    comes out longer than the dummy block.
    """
    weight_of = dict(zip(columns, col_weights, strict=True))
    mapping = get_dummies_mapping(quali, dummy_categorical)
    expanded = [weight_of[col] for col in quanti]
    expanded += [weight_of[col] for col, dummies in mapping.items() for _ in dummies]
    return np.array(expanded, dtype=np.float64)


TYPES = {
    int: "int",
    np.int_: "int",
    float: "float",
    np.float64: "float",
    str: "string",
    np.str_: "string",
    bool: "bool",
    np.bool_: "bool",
}


def get_type_as_string(value: Any) -> str:
    """Return string of type."""
    return TYPES[type(value)]


def get_grouped_modality_values(
    mapping: dict[str, list[str]], to_group: pd.DataFrame
) -> pd.DataFrame:
    """Get the sum of the values of modalities into the category.

    Parameters
    ----------
    mapping :
        mapping between categorical columns and their dummy equivalent
    to_group :
        dataframe from which to sum the values of modalities, which are
        passed as the index.

    Returns:
    -------
        a dataframe with the categorical variables without the dummies
    """
    if not mapping:  # We have no mapping, we have no categorical variables
        return to_group
    grouped_contributions = {}
    for original_col, dummy_columns in mapping.items():
        grouped_contributions[original_col] = to_group.loc[dummy_columns].sum(axis=0)

    grouped_contributions = pd.DataFrame.from_dict(
        data=grouped_contributions,
        orient="index",
        columns=to_group.columns,
    )

    # FIXME: Do not modify the order or columns
    # https://github.com/octopize/saiph/issues/40
    grouped = pd.concat([to_group, grouped_contributions])
    grouped = grouped.drop(concat(mapping.values()))
    return grouped
