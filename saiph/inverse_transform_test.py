from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from saiph.inverse_transform import (
    get_random_weighted_columns,
    inverse_transform,
    undummify,
)
from saiph.projection import fit, fit_transform


@pytest.mark.parametrize(
    "weights, expected_index",
    [
        ([0.3, 0.7, 0.01], 1),
        ([0.7, 0.3, 0.01], 0),
        ([0.01, 0.7, 0.3], 1),
        ([0.01, 0.3, 0.7], 2),
    ],
)
def test_get_random_weighted_columns(weights: List[float], expected_index: int) -> None:
    """Verify that get_random_weighted_columns returns the correct column."""
    df = pd.DataFrame(data=[weights])
    result = get_random_weighted_columns(df, np.random.default_rng(1))
    assert result.values[0] == expected_index


@pytest.mark.parametrize(
    "use_max_modalities, expected",
    [
        (
            True,
            pd.DataFrame(
                [["wrench", "orange"], ["hammer", "apple"]], columns=["tool", "fruit"]
            ),
        ),
        (
            False,
            pd.DataFrame(
                [["wrench", "orange"], ["wrench", "apple"]], columns=["tool", "fruit"]
            ),
        ),
    ],
)
def test_undummify(
    mapping: Dict[str, List[str]], use_max_modalities: bool, expected: pd.DataFrame
) -> None:
    """Test undummify a disjunctive table with different use_max_modalities."""
    dummy_df = pd.DataFrame(
        [[0.3, 0.7, 0.01, 0.99], [0.51, 0.49, 0.8, 0.2]],
        columns=["tool___hammer", "tool___wrench", "fruit___apple", "fruit___orange"],
    )

    df = undummify(dummy_df, mapping, use_max_modalities=use_max_modalities, seed=321)

    assert_frame_equal(df, expected)


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


# using df with more dimensions than individuals and high column weights
# allows for a more balanced probability in modality assignment during inverse transform


def test_inverse_transform_with_ponderation() -> None:
    """Verify that use_max_modalities=False returns a random ponderation of modalities."""
    df = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 3], [2, 2, 10]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    inverse_expected = pd.DataFrame(
        zip(["c", "b", "a"], ["ZZ", "ZZ", "WW"], [1, 2, 2], [4, 4, 4]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    coord, model = fit_transform(df, col_weights=np.array([1, 2000, 1, 1]))
    result = inverse_transform(
        coord, model, use_approximate_inverse=True, use_max_modalities=False, seed=46
    )
    assert_frame_equal(result, inverse_expected)


def test_inverse_transform_deterministic() -> None:
    """Verify that use_max_modalities=True returns a deterministic of modalities."""
    df = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 3], [2, 2, 10]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    inverse_expected = pd.DataFrame(
        zip(["a", "b", "c"], ["ZZ", "ZZ", "WW"], [1, 2, 2], [4, 4, 4]),
        columns=["cat1", "cat2", "cont1", "cont2"],
    )
    coord, model = fit_transform(df, col_weights=np.array([1, 2000, 1, 1]))
    result = inverse_transform(
        coord, model, use_approximate_inverse=True, use_max_modalities=True, seed=46
    )
    assert_frame_equal(result, inverse_expected)


@pytest.mark.skip(
    reason="""Different results on different architectures.
            See https://github.com/octopize/saiph/issues/72"""
)
def test_inverse_from_coord_mca(
    wbcd_quali_df: pd.DataFrame,
    wbcd_supplemental_coord_quali: pd.DataFrame,
) -> None:
    """Check that inverse supplemental coordinates using MCA yield correct results.

    We use `use_max_modalities=False` to keep the data logical.
    We compare indicators of the distributions for each column.
    """
    model = fit(wbcd_quali_df, nf="all")

    reversed_individuals = inverse_transform(
        wbcd_supplemental_coord_quali, model, use_max_modalities=False
    )

    reversed_individuals = reversed_individuals.astype("int")
    wbcd_quali_df = wbcd_quali_df.astype("int")

    reversed_statistics = reversed_individuals.describe()
    wbcd_statistics = wbcd_quali_df.describe()

    assert_series_equal(wbcd_statistics.loc["count"], reversed_statistics.loc["count"])
    assert_allclose(
        wbcd_statistics.loc["mean"], reversed_statistics.loc["mean"], atol=0.25
    )
    assert_allclose(
        wbcd_statistics.loc["std"], reversed_statistics.loc["std"], atol=0.35
    )
    # assert equal for the min as there are many low values
    assert_series_equal(wbcd_statistics.loc["min"], reversed_statistics.loc["min"])
    assert_allclose(wbcd_statistics.loc["25%"], reversed_statistics.loc["25%"], atol=1)
    assert_allclose(wbcd_statistics.loc["50%"], reversed_statistics.loc["50%"], atol=1)
    assert_allclose(wbcd_statistics.loc["75%"], reversed_statistics.loc["75%"], atol=2)
    assert_series_equal(wbcd_statistics.loc["max"], reversed_statistics.loc["max"])


def test_inverse_from_coord_pca(
    wbcd_quanti_df: pd.DataFrame,
    wbcd_supplemental_coord_quanti: pd.DataFrame,
) -> None:
    """Check that inverse supplemental coordinates using PCA yield correct results.

    We use `use_max_modalities=False` to keep the data logical.
    We compare indicators of the distributions for each column.
    """
    model = fit(wbcd_quanti_df, nf="all")

    reversed_individuals = inverse_transform(
        wbcd_supplemental_coord_quanti, model, use_max_modalities=False
    )

    reversed_statistics = reversed_individuals.describe()
    wbcd_statistics = wbcd_quanti_df.describe()

    assert_series_equal(wbcd_statistics.loc["count"], reversed_statistics.loc["count"])
    assert_allclose(
        wbcd_statistics.loc["mean"], reversed_statistics.loc["mean"], atol=0.4
    )
    assert_allclose(
        wbcd_statistics.loc["std"], reversed_statistics.loc["std"], atol=0.7
    )
    # assert equal for the min as there are many low values
    assert_series_equal(wbcd_statistics.loc["min"], reversed_statistics.loc["min"])
    assert_allclose(wbcd_statistics.loc["25%"], reversed_statistics.loc["25%"], atol=1)
    assert_allclose(wbcd_statistics.loc["50%"], reversed_statistics.loc["50%"], atol=1)
    assert_allclose(wbcd_statistics.loc["75%"], reversed_statistics.loc["75%"], atol=1)
    assert_series_equal(
        wbcd_statistics.loc["max"], reversed_statistics.loc["max"], atol=1
    )


def test_inverse_from_coord_famd(
    wbcd_mixed_df: pd.DataFrame,
    wbcd_supplemental_coord_mixed: pd.DataFrame,
) -> None:
    """Check that inverse supplemental coordinates using FAMD yield correct results.

    We use `use_max_modalities=False` to keep the data logical.
    We compare indicators of the distributions for each column.
    """
    model = fit(wbcd_mixed_df, nf="all")
    reversed_individuals = inverse_transform(
        wbcd_supplemental_coord_mixed, model, use_max_modalities=False
    )

    reversed_statistics = reversed_individuals.describe()
    wbcd_statistics = wbcd_mixed_df.describe()

    assert_series_equal(wbcd_statistics.loc["count"], reversed_statistics.loc["count"])
    assert_allclose(
        wbcd_statistics.loc["mean"], reversed_statistics.loc["mean"], atol=0.4
    )
    assert_allclose(
        wbcd_statistics.loc["std"], reversed_statistics.loc["std"], atol=0.6
    )
    # assert equal for the min as there are many low values
    assert_series_equal(wbcd_statistics.loc["min"], reversed_statistics.loc["min"])
    assert_series_equal(wbcd_statistics.loc["25%"], reversed_statistics.loc["25%"])
    assert_series_equal(wbcd_statistics.loc["50%"], reversed_statistics.loc["50%"])
    assert_allclose(wbcd_statistics.loc["75%"], reversed_statistics.loc["75%"], atol=1)
    assert_series_equal(
        wbcd_statistics.loc["max"], reversed_statistics.loc["max"], atol=1
    )
