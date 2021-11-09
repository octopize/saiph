from typing import Any

from numpy.typing import NDArray


def fit_check_params(nf: int, col_w: NDArray[Any], shape_colw: int) -> None:
    if nf <= 0:
        raise ValueError("nf", "The number of components must be positive.")

    if len(col_w) != shape_colw:
        raise ValueError(
            "col_w",
            f"The weight parameter should be of size {str(shape_colw)}.",
        )
