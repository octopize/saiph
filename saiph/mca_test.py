import pandas as pd
from pandas.testing import assert_frame_equal

from saiph.mca import fit


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
            "Dim. 1": [0.7, 0.7],
            "Dim. 2": [0.7, 0.7],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)


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

    result1, _, _ = fit(df)
    result2, _, _ = fit(df_2)

    assert_frame_equal(result1, result2)


def test_fit() -> None:
    df = pd.DataFrame(
        {
            "tool": ["toaster", "hammer"],
            "score": ["aa", "aa"],
        }
    )

    result, _, _ = fit(df)

    expected = pd.DataFrame(
        {
            "Dim. 1": [0.7, -0.7],
            "Dim. 2": [-0.7, -0.7],
        }
    )
    assert_frame_equal(result, expected, check_exact=False, atol=0.01)
