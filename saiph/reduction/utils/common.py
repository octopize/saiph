from itertools import repeat
from typing import Any, Dict, List, OrderedDict, Tuple

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray
from toolz import concat

from saiph.reduction import DUMMIES_PREFIX_SEP


def get_projected_column_names(n: int) -> List[str]:
    return [f"Dim. {i + 1}" for i in range(n)]


def get_uniform_row_weights(n: int) -> NDArray[np.float64]:
    return np.array([k for k in repeat(1 / n, n)])


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


def explain_variance(
    s: NDArray[Any], df: pd.DataFrame, nf: int
) -> Tuple[NDArray[Any], NDArray[Any]]:
    explained_var: NDArray[Any] = ((s**2) / (df.shape[0] - 1))[:nf]
    summed_explained_var = explained_var.sum()
    if summed_explained_var == 0:
        explained_var_ratio: NDArray[np.float_] = np.array([np.nan])
    else:
        explained_var_ratio = explained_var / explained_var.sum()
    return explained_var, explained_var_ratio


def get_modalities_types(df: pd.DataFrame) -> Dict[str, str]:
    modalities_types = {col: get_type_as_string(df.loc[0, col]) for col in df.columns}
    return modalities_types


def get_dummies_mapping(
    columns: List[str], dummy_columns: List[str]
) -> Dict[str, List[str]]:
    """Get mapping between original column and all dummy columns."""
    return OrderedDict(
        {
            col: list(
                filter(
                    lambda c: c.startswith(f"{col}{DUMMIES_PREFIX_SEP}"), dummy_columns
                )
            )
            for col in columns
        }
    )


TYPES = {
    int: "int",
    float: "float",
    str: "string",
    bool: "bool",
    np.int_: "int",
    np.float_: "float",
}


def get_type_as_string(value: Any) -> str:
    """Return string of type."""
    return TYPES[type(value)]


def get_grouped_modality_values(
    mapping: Dict[str, List[str]], to_group: pd.DataFrame
) -> pd.DataFrame:
    """Get the sum of the values of modalities into the category.

    Parameters
    ----------
    mapping :
        mapping between categorical columns and their dummy equivalent
    to_group :
        dataframe from which to sum the values of modalities

    Returns
    -------
        a dataframe with the categorical variables without the dummies
    """
    grouped_contributions = {}
    for original_col, dummy_columns in mapping.items():
        grouped_contributions[original_col] = to_group.loc[dummy_columns].sum(axis=0)

    grouped_contributions = pd.DataFrame.from_dict(
        data=grouped_contributions,
        orient="index",
        columns=to_group.columns,
    )

    to_group = pd.concat([to_group, grouped_contributions])
    to_group = to_group.drop(concat(mapping.values()))
    return to_group
