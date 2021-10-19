import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

from saiph.pca import fit, center, scaler


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
            "Dim. 0": [-1.41, 1.41],
            "Dim. 1": [0.0, 0.0],
        }
    )
    expected_v = np.array([[0.71, 0.71], [-0.71, 0.71]])
    expected_explained_var = np.array([1., 0.])
    expected_explained_var_ratio = np.array([1., 0.])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [2., 3.])
    assert_allclose(model.std, [1., 1.])


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
            "Dim. 0": [1.94, -0.28, -1.66],
            "Dim. 1": [-0.07, 0.18, -0.11],
        }
    )
    expected_v = np.array([[-0.544914, -0.838492], [-0.838492,  0.544914]])
    expected_explained_var = np.array([0.367571, 0.002799])
    expected_explained_var_ratio = np.array([1., 0.])

    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    assert_allclose(model.explained_var_ratio, expected_explained_var_ratio, atol=0.01)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [2., 3.666667])
    # Default value when not scaling
    assert_allclose(model.std, 0.)


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
            "Dim. 0": [0.0, 0.0],
            "Dim. 1": [0.0, 0.0],
        }
    )
    expected_v = np.array([[1., 0.], [0., 1.]])
    expected_explained_var = np.array([0., 0.])
    
    assert_frame_equal(result, expected_result, check_exact=False, atol=0.01)

    assert_frame_equal(model.df, df)
    assert_allclose(model.V, expected_v, atol=0.01)
    assert_allclose(model.explained_var, expected_explained_var, atol=0.01)
    # np.nan == np.nan returns False
    assert pd.isnull(model.explained_var_ratio)
    assert_allclose(model.variable_coord, model.V.T, atol=0.01)
    assert_allclose(model.mean, [1., 2.])
    assert_allclose(model.std, [1., 1.])



    
# TODO: Unify center and scale
# Return type should be a parameter

def test_center_scaler() -> None:
    df = pd.DataFrame(
        {
            0: [1.0, 12.0],
            1: [2.0, 4.0],
        }
    )

    _, model, _ = fit(df, scale=True)

    print('original')
    print(model.df)
    print('Center')
    df1, mean, std = center(model.df.copy(), scale=True)
    print(df1)
    print("type1")
    print(df1.dtypes)
    #print(mean)
    #print(std)
    print('scaler')
    df2 = scaler(model, None)
    print(df2)

    assert_frame_equal(df1, pd.DataFrame(df2), check_column_type=False, check_names=False)
