import pytest

from saiph.models import get_number_of_dimensions_from_categorical_and_continuous


@pytest.mark.parametrize(
    "dummy_categorical,original_continuous,expected_result",
    [
        (10, 5, 14),
        (2, 3, 4),
    ],
)
def test_get_number_of_dimensions_from_categorical_and_continuous(
    dummy_categorical: int, original_continuous: int, expected_result: int
) -> None:
    assert (
        get_number_of_dimensions_from_categorical_and_continuous(
            dummy_categorical=list(range(dummy_categorical)),
            original_continuous=list(range(original_continuous)),
        )
        == expected_result
    )
