import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from saiph.reduction.famd import center, fit, scaler
from saiph.reduction.pca import center as center_pca
from saiph.reduction.pca import fit as fit_pca
from saiph.reduction.pca import scaler as scaler_pca


def test_fit_mix() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
        }
    )

    result, model, _ = fit(df)

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
    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.s, expected_s, atol=0.01)
    assert_allclose(model.U, expected_u, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01),
    assert_allclose(model.variable_coord, model.V.T)
    assert np.array_equal(
        model._modalities, ["tool_hammer", "tool_toaster", "score_aa", "score_ab"]
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
    print(model._modalities)
    assert np.array_equal(
        model._modalities, ["tool_hammer", "tool_toaster", "score_aa", "score_ab"]
    )


def test_fit_zero() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "toaster"],
            "score": ["aa", "aa"],
        }
    )

    result, _, _ = fit(df)

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

    _, model, param = fit(original_df)
    df = scaler(model, param)

    _, model_pca, _ = fit_pca(original_df[param.quanti])
    df_pca = scaler_pca(model_pca)

    assert_frame_equal(df[param.quanti], df_pca[param.quanti])


def test_center_pca_famd() -> None:
    original_df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "ab"],
            "size": [1.0, 4.0],
            "age": [55, 62],
        }
    )

    _, _, param = fit(original_df)
    df, mean1, std1, _, _ = center(original_df, quali=param.quali, quanti=param.quanti)

    df_pca, mean2, std2 = center_pca(original_df[param.quanti])

    assert_frame_equal(df[param.quanti], df_pca[param.quanti])

    assert_series_equal(mean1, mean2)
    assert_series_equal(std1, std2)
