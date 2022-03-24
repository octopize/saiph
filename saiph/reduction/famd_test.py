import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction import DUMMIES_PREFIX_SEP
from saiph.reduction.famd import center, fit_transform, scaler, transform
from saiph.reduction.pca import center as center_pca
from saiph.reduction.pca import fit_transform as fit_pca
from saiph.reduction.pca import scaler as scaler_pca


def test_fit_mix() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
        }
    )

    result, model = fit_transform(df)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.73, 1.73],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v = np.array(
        [
            [0.57735, 0.408248, -0.408248, -0.408248, 0.408248],
            [0.816497, -0.288675, 0.288675, 0.288675, -0.288675],
        ]
    )
    expected_s = np.array([1.224745e00, 0.0])
    expected_u = np.array([[-1.0, 1.0], [1.0, 1.0]])
    expected_explained_var = np.array([1.5, 0.0])
    expected_explained_var_ratio = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.s, expected_s, atol=0.01)
    assert_allclose(model.U, expected_u, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    assert_allclose(model.variable_coord, model.V.T)
    assert np.array_equal(
        model._modalities,
        [
            f"tool{DUMMIES_PREFIX_SEP}hammer",
            f"tool{DUMMIES_PREFIX_SEP}toaster",
            f"score{DUMMIES_PREFIX_SEP}aa",
            f"score{DUMMIES_PREFIX_SEP}ab",
        ],
    )
    # Pertinent ?
    # assert_allclose(model.D_c,np.array([[2.0, 0.0, 0.0],
    # [0.0, 2.0, 0.0], [0.0, 0.0, 1.41421356]]), atol=0.01)
    assert model.D_c is None

    assert_allclose(model.mean, np.array(2.5))
    assert_allclose(model.std, np.array(1.5))

    assert_series_equal(
        model.prop.reset_index(drop=True),
        pd.Series([0.5, 0.5, 0.5, 0.5]).reset_index(drop=True),
        atol=0.01,
    )
    assert np.array_equal(
        model._modalities,
        [
            f"tool{DUMMIES_PREFIX_SEP}hammer",
            f"tool{DUMMIES_PREFIX_SEP}toaster",
            f"score{DUMMIES_PREFIX_SEP}aa",
            f"score{DUMMIES_PREFIX_SEP}ab",
        ],
    )


def test_transform() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )

    _, model = fit_transform(df)

    df_transformed = transform(df, model)
    df_expected = pd.DataFrame(
        {
            "Dim. 1": [2.0, -2],
            "Dim. 2": [0.0, 0],
        }
    )

    assert_frame_equal(df_transformed, df_expected)


def test_transform_vs_coord() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )

    coord, model = fit_transform(df)
    df_transformed = transform(df, model)

    assert_frame_equal(df_transformed, coord)


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, _ = fit_transform(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


def test_scaler_pca_famd() -> None:
    original_df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )

    _, model = fit_transform(original_df)
    df = scaler(model, original_df)

    _, model_pca = fit_pca(original_df[model.original_continuous])
    df_pca = scaler_pca(model_pca, original_df)

    assert_frame_equal(df[model.original_continuous], df_pca[model.original_continuous])


def test_center_pca_famd() -> None:
    original_df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )

    _, model = fit_transform(original_df)
    continous = model.original_continuous
    categorical = model.original_categorical
    df, mean1, std1, _, _ = center(original_df, quali=categorical, quanti=continous)

    df_pca, mean2, std2 = center_pca(original_df[continous])

    assert_frame_equal(df[continous], df_pca[continous])

    assert_series_equal(mean1, mean2)
    assert_series_equal(std1, std2)
