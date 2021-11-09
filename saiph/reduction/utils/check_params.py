from typing import Any, List


def fit_check_params(nf: int, col_w: List[Any], shape_colw: int) -> None:
    if nf <= 0:
        raise ValueError("nf", "The number of components must be positive.")

    if len(col_w) != shape_colw:
        raise ValueError(
            "col_w",
            f"The weight parameter should be of size {str(shape_colw)}.",
        )
