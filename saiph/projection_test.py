from typing import List
from unittest import result

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from saiph.projection import (
    fit_transform,
    get_dummies_mapping,
    get_random_weighted_columns,
    inverse_transform,
    stats,
    transform,
    inverse_transform
)
from saiph.reduction import DUMMIES_PREFIX_SEP


def test_transform_then_inverse_FAMD(iris_df: pd.DataFrame) -> None:
    transformed, model = fit_transform(iris_df, nf="all")
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, iris_df)


def test_transform_then_inverse_PCA(iris_quanti_df: pd.DataFrame) -> None:
    transformed, model = fit_transform(iris_quanti_df, nf="all")
    un_transformed = inverse_transform(transformed, model)
    assert_frame_equal(un_transformed, iris_quanti_df)


def test_transform_then_inverse_MCA(quali_df: pd.DataFrame) -> None:

    df = quali_df
    transformed, model = fit_transform(df)
    un_transformed = inverse_transform(transformed, model)
    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_type(quali_df: pd.DataFrame) -> None:
    df = quali_df

    df = df.astype("object")
    transformed, model = fit_transform(df)
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_FAMD_weighted(mixed_df: pd.DataFrame) -> None:
    df = mixed_df
    transformed, model = fit_transform(df, col_w=np.array([2, 3]))
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_PCA_weighted(quanti_df: pd.DataFrame) -> None:
    df = quanti_df
    coords, model = fit_transform(df, col_w=np.array([2, 1, 3]))
    un_transformed = inverse_transform(coords, model)

    assert_frame_equal(un_transformed, df)


def test_transform_then_inverse_MCA_weighted() -> None:
    df = pd.DataFrame(
        {
            "variable_1": ["1", "3", "3", "3", "1", "2", "2", "1", "1", "2"],
            "variable_2": ["1", "1", "1", "2", "2", "1", "1", "1", "2", "2"],
            "variable_3": ["1", "2", "1", "2", "1", "2", "1", "1", "2", "2"],
            "variable_4": [
                "red",
                "blue",
                "blue",
                "green",
                "red",
                "blue",
                "red",
                "red",
                "red",
                "red",
            ],
        }
    )

    transformed, model = fit_transform(df, col_w=np.array([2, 1, 3, 2]))
    un_transformed = inverse_transform(transformed, model)

    assert_frame_equal(un_transformed, df)


def test_coords_vs_transform_with_multiple_nf(iris_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        fit_transform(iris_df, nf=10000)

    with pytest.raises(ValueError):
        fit_transform(iris_df, nf=-1)

    for n in range(7):
        coord, model = fit_transform(iris_df, nf=n)
        transformed = transform(iris_df, model)
        assert_frame_equal(coord, transformed)


df_pca = pd.DataFrame(
    {
        0: [
            1000.0,
            3000.0,
            10000.0,
            1500.0,
            700.0,
            3300.0,
            5000.0,
            2000.0,
            1200.0,
            6000.0,
        ],
        1: [185.0, 174.3, 156.8, 182.7, 180.3, 179.2, 164.7, 192.5, 191.0, 169.2],
        2: [1, 5, 10, 2, 4, 4, 7, 3, 1, 6],
    }
)

df_famd = pd.DataFrame(
    {
        "variable_1": [4, 5, 6, 7, 11, 2, 52],
        "variable_2": [10, 20, 30, 40, 10, 74, 10],
        "variable_3": ["red", "blue", "blue", "green", "red", "blue", "red"],
        "variable_4": [100, 50, -30, -50, -19, -29, -20],
    }
)

df_mca = pd.DataFrame(
    {
        0: [
            "red",
            "red",
            "whithe",
            "whithe",
            "red",
            "whithe",
            "red",
            "red",
            "whithe",
            "red",
        ],
        1: [
            "beef",
            "chicken",
            "fish",
            "fish",
            "beef",
            "chicken",
            "beef",
            "chicken",
            "fish",
            "beef",
        ],
        2: [
            "france",
            "espagne",
            "france",
            "italy",
            "espagne",
            "france",
            "france",
            "espagne",
            "chine",
            "france",
        ],
    }
)


@pytest.mark.parametrize(
    "df_input,expected_type", [(df_pca, "pca"), (df_mca, "mca"), (df_famd, "famd")]
)
def test_eval(df_input, expected_type):
    _, model = fit_transform(df_input)
    assert model.type == expected_type


# check contribution of variables to dim

expected_pca_contrib = [32.81178064277649, 33.12227570926467, 34.065943647958846]
expected_mca_contrib = [
    13.314231201732547,
    19.971346802598834,
    7.924987451762375,
    2.8647115861394203,
    24.435070805047193,
    11.119305005676665,
    8.778349565191498,
    0.47269257617482885,
    11.119305005676662,
]
expected_famd_contrib = [
    15.696161557629662,
    36.08406414786589,
    11.420291290785196,
    9.852860955848701,
    6.104123324745425,
    20.842498723125182,
]


@pytest.mark.parametrize(
    "df_input,expected_contrib",
    [
        (df_pca, expected_pca_contrib),
        (df_mca, expected_mca_contrib),
        (df_famd, expected_famd_contrib),
    ],
)
def test_var_contrib(df_input, expected_contrib):
    _, model = fit_transform(df_input)
    stats(model, df_input)
    assert_allclose(model.contributions["Dim. 1"], expected_contrib, atol=1e-07)


# check cor of variables to dim (if cor is ok so is cos2)

expected_pca_cor = [0.9621683005738202, -0.9667100394109722, 0.9803843201246043]
expected_mca_cor = [
    -0.9169410964961192,
    0.9169410964961192,
    -0.5776131247092218,
    -0.3215176498417082,
    0.9390120540605189,
    0.5586386162833192,
    -0.5628216589197227,
    -0.15453176858793358,
    0.5586386162833191,
]
expected_famd_cor = [
    -0.5930925452224494,
    0.8992562362632682,
    -0.5058995881660323,
    0.6216213705726737,
    0.399494477072544,
    -0.9041066243572436,
]


@pytest.mark.parametrize(
    "df_input,expected_cor",
    [
        (df_pca, expected_pca_cor),
        (df_mca, expected_mca_cor),
        (df_famd, expected_famd_cor),
    ],
)
def test_var_cor(df_input, expected_cor):
    _, model = fit_transform(df_input)
    stats(model, df_input)
    assert_allclose(model.correlations["Dim. 1"], expected_cor, atol=1e-07)


# Check percentage of explained variance

expected_pca_explained_var_ratio = [
    0.9404831846910865,
    0.040151017139633684,
    0.01936579816927985,
]
expected_mca_explained_var_ratio = [
    0.42099362789799255,
    0.23253662367291794,
    0.1666666666666665,
    0.1314458557658021,
    0.035220810900864555,
]
expected_famd_explained_var_ratio = [
    0.44820992177856206,
    0.2801754650846534,
    0.1707856006031922,
    0.05566366403780453,
    0.04516534849578783,
]


@pytest.mark.parametrize(
    "df_input,expected_var_ratio",
    [
        (df_pca, expected_pca_explained_var_ratio),
        (df_mca, expected_mca_explained_var_ratio),
        (df_famd, expected_famd_explained_var_ratio),
    ],
)
def test_var_ratio(df_input, expected_var_ratio) -> None:
    _, model = fit_transform(df_input)
    stats(model, df_input)
    assert_allclose(model.explained_var_ratio[0:5], expected_var_ratio, atol=1e-07)


def test_get_dummies_mapping(quali_df: pd.DataFrame) -> None:
    dummy_columns = pd.get_dummies(quali_df, prefix_sep=DUMMIES_PREFIX_SEP).columns

    result = get_dummies_mapping(quali_df.columns, dummy_columns)

    sep = DUMMIES_PREFIX_SEP
    expected = {
        "tool": [f"tool{sep}hammer", f"tool{sep}wrench"],
        "fruit": [f"fruit{sep}apple", f"fruit{sep}orange"],
    }
    assert result == expected


@pytest.mark.parametrize(
    "weights, expected_index",
    [
        ([0.3, 0.7, 0.01], 1),
        ([0.7, 0.3, 0.01], 0),
        ([0.01, 0.7, 0.3], 1),
        ([0.01, 0.3, 0.7], 2),
    ],
)
def test_get_random_weighted_columns(weights: List[float], expected_index: int):
    df = pd.DataFrame(data=[weights])
    result = get_random_weighted_columns(df, np.random.default_rng(1))
    assert result.values[0] == expected_index


# wider than len df
def test_inverse_transform_raises_value_error_when_wider_than_df() -> None:
    wider_df = pd.DataFrame(
        {
            "variable_1": ["a", "b", "c"],
            "variable_2": ["ZZ", "ZZ", "WW"],
        }
    )
    coord, model = fit_transform(wider_df)
    with pytest.raises(ValueError, match=r"n_dimensions"):
        inverse_transform(coord, model)

# using df with more dimension than N and high weight
# allow more balanced probability inmodality assignment during inverse transform

def test_inverse_transform_with_ponderation() -> None:
    df = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 3], [2, 2, 10]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    inverse_expected = pd.DataFrame(
        zip(["c", "b", "a"], ["ZZ", "ZZ", "WW"], [1, 2, 2], [4, 4, 4]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    coord, model = fit_transform(df, col_w=[1, 2000,1,1])
    result = inverse_transform(coord, model, use_approximate_inverse=True, use_max_modalities=False, seed=46)
    assert_frame_equal(result, inverse_expected)

def test_inverse_transform_deterministic() -> None:
    df = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 3], [2, 2, 10]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    inverse_expected = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 2], [4, 4, 4]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    coord, model = fit_transform(df, col_w=[1, 2000,1,1])
    result = inverse_transform(coord, model, use_approximate_inverse=True, use_max_modalities=True, seed=46)
    assert_frame_equal(result, inverse_expected)


