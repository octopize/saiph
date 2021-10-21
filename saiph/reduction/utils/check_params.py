def fit_check_params(nf, col_w, shape_colw):
    if nf <= 0:
        raise ValueError("nf", "The number of components must be positive.")

    if len(col_w) != shape_colw:
        raise ValueError(
            "col_w",
            f"The weight parameter should be of size {str(shape_colw)}.",
        )
