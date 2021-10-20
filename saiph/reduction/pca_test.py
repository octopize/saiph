import numpy as np
import pandas as pd
import prince
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from sklearn import decomposition

from saiph.reduction.pca import center, fit, scaler, transform


def test_fit_scale() -> None:
    scale = True

    df = pd.DataFrame(
        {
            "one": [1.0, 3.0],
            "two": [2.0, 4.0],
        }
    )

    result, model, _ = fit(df, scale=scale)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [-1.41, 1.41],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v = np.array([[0.71, 0.71], [-0.71, 0.71]])
    expected_explained_var = np.array([1.0, 0.0])
    expected_explained_var_ratio = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [2.0, 3.0])
    assert_allclose(model.std, [1.0, 1.0])


def test_fit_not_scale() -> None:
    scale = False

    df = pd.DataFrame(
        {
            "one": [1.0, 2.0, 3.0],
            "two": [2.0, 4.0, 5.0],
        }
    )

    result, model, _ = fit(df, scale=scale)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [1.94, -0.28, -1.66],
            "Dim. 2": [-0.07, 0.18, -0.11],
        }
    )
    expected_v = np.array([[-0.544914, -0.838492], [-0.838492, 0.544914]])
    expected_explained_var = np.array([0.367571, 0.002799])
    expected_explained_var_ratio = np.array([1.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [2.0, 3.666667])
    # Default value when not scaling
    assert_allclose(model.std, 0.0)


def test_fit_zero() -> None:
    scale = True

    df = pd.DataFrame(
        {
            "one": [1, 1],
            "two": [2, 2],
        }
    )

    result, model, _ = fit(df, scale=scale)

    expected_result = pd.DataFrame(
        {
            "Dim. 1": [0.0, 0.0],
            "Dim. 2": [0.0, 0.0],
        }
    )
    expected_v = np.array([[1.0, 0.0], [0.0, 1.0]])
    expected_explained_var = np.array([0.0, 0.0])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    # np.nan == np.nan returns False
    assert pd.isnull(model.explained_var_ratio)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [1.0, 2.0])
    assert_allclose(model.std, [1.0, 1.0])


# TODO: Unify scaler/center one day ?
def test_center_scaler() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    _, model, _ = fit(df, scale=True)

    print("original")
    print(model.df)
    print("Center")
    df1, mean, std = center(model.df.copy(), scale=True)
    print(df1)
    print("type1")
    print(df1.dtypes)
    # print(mean)
    # print(std)
    print("scaler")
    df2 = scaler(model, None)
    print(df2)

    assert_frame_equal(df1, df2)


def test_transform() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    _, model, param = fit(df, scale=True)

    df_transformed = transform(df, model, param)

    expected_transformed = pd.DataFrame(
        {
            "Dim. 1": [-1.414214, 1.414214],
            "Dim. 2": [0.0, 0.0],
        }
    )

    assert_frame_equal(df_transformed, expected_transformed, atol=0.01)


def test_compare_sklearn_simple() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    coord, _, _ = fit(df, scale=False)
    pca = decomposition.PCA(n_components=2)
    coord_scikit = pca.fit_transform(df)

    assert_allclose(coord.to_numpy(), coord_scikit, atol=0.0001)


def test_compare_sklearn_full(iris_quanti_df) -> None:
    coord, _, _ = fit(iris_quanti_df, scale=False)
    pca = decomposition.PCA(n_components=4)
    coord_scikit = pca.fit_transform(iris_quanti_df)

    assert_allclose(coord.to_numpy(), coord_scikit, atol=0.0001)


def test_compare_sklearn_nf_reduced(iris_quanti_df) -> None:
    coord, _, _ = fit(iris_quanti_df, scale=False, nf=3)
    pca = decomposition.PCA(n_components=3)
    coord_scikit = pca.fit_transform(iris_quanti_df)

    assert_allclose(coord.to_numpy(), coord_scikit, atol=0.0001)


def test_compare_prince_full(iris_quanti_df) -> None:
    pca = prince.PCA(n_components=4, rescale_with_std=False)
    pca = pca.fit(iris_quanti_df)
    pca = pca.transform(iris_quanti_df)

    coord, _, _ = fit(iris_quanti_df, scale=False)

    assert_allclose(coord.to_numpy(), pca, atol=0.0001)
