import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.mca import (
    fit,
    fit_transform,
    get_variable_contributions,
    transform,
)
from saiph.reduction.utils.common import get_projected_column_names


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, model = fit_transform(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.7, -0.7],
            "Dim. 2": [-0.7, -0.7],
        }
    )
    expected_v: NDArray[np.float_] = np.array(
        [[-0.707107, 0.707107, -0.0], [-0.707107, -0.707107, 0.0]]
    )
    expected_explained_var: NDArray[np.float_] = np.array([1.25000e-01, 3.85186e-34])
    expected_explained_var_ratio: NDArray[np.float_] = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    assert np.array_equal(
        model._modalities,  # type: ignore
        [
            f"tool{DUMMIES_PREFIX_SEP}hammer",
            f"tool{DUMMIES_PREFIX_SEP}toaster",
            f"score{DUMMIES_PREFIX_SEP}aa",
        ],
    )
    assert_allclose(
        model.D_c,
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.41421356]]),
        atol=0.01,
    )
    assert model.mean is None
    assert model.std is None


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, model = fit_transform(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.7, 0.7],
            "Dim. 2": [0.7, 0.7],
        }
    )
    expected_v: NDArray[np.float_] = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_explained_var: NDArray[np.float_] = np.array([0.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert pd.isna(model.explained_var_ratio).all()
    assert np.array_equal(
        model._modalities,  # type: ignore
        [f"tool{DUMMIES_PREFIX_SEP}toaster", f"score{DUMMIES_PREFIX_SEP}aa"],
    )
    assert_allclose(
        model.D_c,
        np.array([[1.414214, 0.0], [0.0, 1.414214]]),
        atol=0.01,
    )
    assert model.mean is None
    assert model.std is None


def test_fit_zero_same_df() -> None:
    """Verify that we get the same result if the pattern matches."""
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )
    df_2 = pd.DataFrame(
        {
            "tool": ["hammer", "hammer"],
            "score": ["bb", "bb"],
        }
    )

    result1, model1 = fit_transform(df)
    result2, model2 = fit_transform(df_2)

    assert_frame_equal(result1, result2)

    for k in [
        "explained_var",
        "variable_coord",
        "variable_coord",
        "U",
        "s",
        "mean",
        "std",
        "prop",
        "D_c",
    ]:  # removed "_modalities", "df", "explained_var_ratio"
        k1 = getattr(model1, k)
        k2 = getattr(model2, k)
        if isinstance(k1, pd.DataFrame):
            assert k1.equals(k2)
        elif isinstance(k1, np.ndarray):
            assert np.array_equal(k1, k2)
        else:
            assert k1 == k2

# @pytest.mark.parametrize(
#     "drop_first, expected",
#     [
#         True, expected_transform = pd.DataFrame(
#         {
#             "Dim. 1": [0.707107, 0.707107],
#             "Dim. 2": [0.707107, 0.707107],
#         }
#     )
#         False, expected_transform = pd.DataFrame(
#         {
#             "Dim. 1": [0.707107, 0.707107],
#             "Dim. 2": [0.707107, 0.707107],
#         }
#     )
#     ],
# )
def test_transform_simple() -> None:
    """Verify that mca.transform returns correct output."""
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )
    df_transformed, _ = fit_transform(df, drop_first=False)

    expected_transform = pd.DataFrame(
        {
            "Dim. 1": [0.707107, 0.707107],
            "Dim. 2": [0.707107, 0.707107],
        }
    )

    assert_frame_equal(
        df_transformed, expected_transform, check_exact=False, atol=0.00001
    )

@pytest.mark.parametrize(
    "drop_first",
    [True, False],
)
def test_fit_transform_has_same_output_as_transform(drop_first: bool) -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )
    coord, model = fit_transform(df, drop_first=drop_first)
    df_transformed = transform(df, model, drop_first=drop_first)
    assert_frame_equal(coord, df_transformed)


def test_get_variable_contributions(quali_df: pd.DataFrame) -> None:
    """Verify that mca.get_variable_contributions returns correct output."""
    df = quali_df
    _, model = fit_transform(df, nf=3)

    contributions = get_variable_contributions(model, df, drop_first=False, explode=True)

    expected_contributions = pd.DataFrame.from_dict(
        data={
            "tool___hammer": [25.0, 25.0, 48.104419],
            "tool___wrench": [25.0, 25.0, 48.104419],
            "fruit___apple": [12.5, 12.5, 2.843371],
            "fruit___orange": [37.5, 37.5, 0.947790],
        },
        dtype=np.float_,
        orient="index",
        columns=get_projected_column_names(3),
    )

    # We only look at n - 1 first dimensions because of roundoff errors.
    # https://github.com/octopize/saiph/issues/58
    assert_frame_equal(contributions.iloc[:, :-1], expected_contributions.iloc[:, :-1])


def test_get_variable_contributions_exploded_parameter(mixed_df: pd.DataFrame) -> None:
    """Verify argument explode=False and explode=True in get_variable_contributions.

    Make sure that explode=False is the sum of explode=True for categorical variables.
    """
    df = mixed_df
    variable = "tool"
    _, model = fit_transform(df, nf=3)

    contributions_exploded = get_variable_contributions(model, df, drop_first=False, explode=True)
    contributions_not_exploded = get_variable_contributions(model, df, drop_first=False, explode=False)
    dummies = filter(
        lambda name: f"{variable}{DUMMIES_PREFIX_SEP}" in name,
        contributions_exploded.index,
    )
    assert_series_equal(
        contributions_exploded.loc[list(dummies)].sum(),
        contributions_not_exploded.loc[variable],
        check_names=False,
    )


def test_get_variable_contributions_sum_is_100_with_col_weights_random_mca(
    quali_df: pd.DataFrame,
) -> None:
    model = fit(quali_df, col_weights=[3.0, 2.0], drop_first=False)  # type: ignore
    contributions = get_variable_contributions(model, quali_df, drop_first=False)
    summed_contributions = contributions.sum(axis=0)
    assert_series_equal(summed_contributions, pd.Series([100.0] * 4), check_index=False)


def test_fit_reconstruct() -> None:
    df = pd.read_csv("/Users/olivier/dev/avatar/core/fixtures/wbcd.csv").astype("category")
    
    # df = pd.DataFrame(
    #     {
    #         "tool": ["toaster", "hammer", "hammer", "hammer", "toaster"],
    #         "score": ["aa", "aa", "bb", "bb", "bb"],
    #     }
    # )  # PB !!!

    # df = pd.DataFrame(
    #     {
    #         "var1": ["0", "1", "0", "0", "0", "1", "0", "1", "0", "0", "0", "1"],
    #         # "var1bis": ["1", "0", "1", "1", "1", "0", "1", "0", "1", "1", "1", "0"],
    #         "score": ["truc", "truc2", "truc3", "truc3", "truc", "truc4", "truc", "truc2", "truc3", "truc3", "truc", "truc4"],
    #     }
    # )  # TEST

    # df = pd.DataFrame(
    #     {
    #         "var1": ["0", "1", "0", "0", "0", "1"],
    #         "var1bis": ["1", "0", "1", "1", "1", "0"],
    #         "score": ["truc", "truc2", "truc3", "truc3", "truc", "truc4"],
    #     }
    # )

    # df = pd.DataFrame(
    #     {
    #         "tool": ["toaster", "toaster", "hammer", "hammer", "toaster"],
    #         "score": ["aa", "aa", "bb", "bb", "bb"],
    #     }
    # )

    # df = pd.DataFrame(
    #     {
    #         "tool": ["toaster", "hammer", "toaster", "hammer", "toaster", "hammer", "hammer", "mouse", "mouse", "mouse"],
    #         "score": ["aa", "aa", "aa", "bb", "aa", "cc", "bb", "cc", "bb", "cc"],
    #         "utility": ["zz", "zz", "zz", "yy", "zz", "yy", "yy", "zz", "yy", "zz"],
    #     }
    # )

    # df = pd.DataFrame(
    #     {
    #         "tool": ["toaster", "hammer", "toaster", "hammer", "toaster", "hammer", "hammer"],
    #         "score": ["aa", "aa", "aa", "bb", "aa", "bb", "bb"],
    #     }
    # )

    result, model = fit_transform(df, drop_first=True)

    print('result: ', result)
    from saiph.inverse_transform import inverse_transform

    from saiph.visualization import plot_circle, plot_var_contribution, plot_projections
    from saiph import stats


    plot_projections(model, df, (0, 1))

    model = stats(model, df)

    print('model.contributions:', model.contributions)
    print('model.explained_var:', model.explained_var)
    print('model.explained_var_ratio:', model.explained_var_ratio)
    plot_circle(model=model, max_var=10)

    plot_var_contribution(
    model.contributions["Dim. 1"].to_numpy(), model.contributions.index.to_numpy()
    )
    plot_var_contribution(
    model.contributions["Dim. 2"].to_numpy(), model.contributions.index.to_numpy()
    )

    inv_trans = inverse_transform(result, model)
    print('inv_trans: ', inv_trans)
    print('SHOULD BE: ', df)

    assert_frame_equal(df, inv_trans)
    
    