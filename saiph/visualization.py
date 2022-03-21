"""Visualization functions."""
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # type: ignore
from numpy.typing import NDArray

from saiph import transform
from saiph.models import Model, Parameters

matplotlib.use("TkAgg", force=True)  # Default matplotlib backend is not GUI compatible


def plot_circle(
    model: Model,
    param: Parameters,
    dimensions: Optional[List[int]] = None,
    min_cor: float = 0.1,
    max_var: int = 7,
) -> None:
    """Plot correlation circle.

    Parameters
    ----------
    model: Model
        The model for transforming new data.
    param: Parameters
        The parameters for transforming new data.
    dimensions: Optional[List[int]]
        Dimensions to help by each axis
    min_cor: float
        Minimuim correlation threshold to display arrow
    max_var: int
        Number of variables to display (in descending order)
    """
    # make sure stats have been computed prior to visualization
    if param.cor is None:
        raise ValueError(
            "empty param, run fit function to create Model class and Parameters class objects"
        )

    # Dimensions start from 1

    # Plotting circle
    dimensions = dimensions or [1, 2]
    figure_axis_size = 6
    explained_var_ratio = model.explained_var_ratio

    circle1 = plt.Circle((0, 0), radius=1, color="k", fill=False)
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    fig.gca().add_artist(circle1)

    # Order dataframe
    cor = param.cor.copy()
    cor["sum"] = cor.apply(
        lambda x: abs(x[dimensions[0] - 1]) + abs(x[dimensions[1] - 1]), axis=1
    )
    cor.sort_values(by="sum", ascending=False, inplace=True)

    # Plotting arrows
    texts = []
    i = 0
    for name, row in cor.iterrows():
        if i < max_var and (
            np.abs(row[dimensions[0] - 1]) > min_cor
            or np.abs(row[dimensions[1] - 1]) > min_cor
        ):
            x = row[dimensions[0] - 1]
            y = row[dimensions[1] - 1]
            plt.arrow(
                0.0,
                0.0,
                x,
                y,
                color="k",
                length_includes_head=True,
                head_width=0.05,
            )

            plt.plot([0.0, x], [0.0, y], "k-")
            texts.append(plt.text(x, y, name, fontsize=2 * figure_axis_size))
            i += 1

    # Plotting vertical lines
    plt.plot([-1.1, 1.1], [0, 0], "k--")
    plt.plot([0, 0], [-1.1, 1.1], "k--")

    # Setting limits and title
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.title("Correlation Circle", fontsize=figure_axis_size * 3)

    plt.xlabel(
        "Dim "
        + str(dimensions[0])
        + " (%s%%)" % str(explained_var_ratio[dimensions[0] - 1] * 100)[:4],
        fontsize=figure_axis_size * 2,
    )
    plt.ylabel(
        "Dim "
        + str(dimensions[1])
        + " (%s%%)" % str(explained_var_ratio[dimensions[1] - 1] * 100)[:4],
        fontsize=figure_axis_size * 2,
    )


def plot_var_contribution(
    values: NDArray[np.float_],
    names: NDArray[np.string_],
    title: str = "Variables contributions",
) -> None:
    """Plot the variable contributions for a given dimension."""
    # plot
    plt.figure(figsize=(12, 6))
    indices = range(len(values))
    plt.bar(indices, values, align="center")
    plt.xticks(indices, names, rotation="horizontal")

    # setting labels and title
    plt.title(title)
    plt.ylabel("Importance")
    plt.xlabel("Variables")
    plt.show()


def plot_explained_var(
    model: Model, max_dims: int = 10, cumulative: bool = False
) -> None:
    """Plot explained variance per dimension.

    Parameters
    ----------
    model: Model
        Model computed by fit.
    max_dims: int
        Maximum number of dimensions to plot
    """
    # explained_percentage

    explained_percentage: NDArray[np.float_] = (
        np.cumsum(model.explained_var_ratio)
        if cumulative
        else model.explained_var_ratio
    )

    if len(explained_percentage) > max_dims:
        explained_percentage = explained_percentage[:max_dims]

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(explained_percentage)), explained_percentage * 100, align="center"
    )
    plt.xticks(
        range(len(explained_percentage)),
        range(1, len(explained_percentage) + 1),
        rotation="horizontal",
    )

    # setting labels and title
    plt.title("Explained variance plot")
    plt.ylabel("Percentage of explained variance")
    plt.xlabel("Dimensions")
    plt.show()


def plot_projections(
    model: Model, param: Parameters, data: pd.DataFrame, dim: Tuple[int, int] = (0, 1)
) -> None:
    """Plot projections in reduced space for input data.

    Parameters
    ----------
    model
        Model computed by fit.
    param
        The parameters for transforming the data.
    data
        Data to plot in the reduced space
    dim
        Axes to use for the 2D plot (default (0,1))
    """
    dim_x, dim_y = dim

    transformed_data = transform(data, model, param)

    # Retrieve column names matching the selected dimensions
    x_name = transformed_data.columns[dim_x]
    y_name = transformed_data.columns[dim_y]

    # Retrieve data
    x = transformed_data[x_name]
    y = transformed_data[y_name]

    # Set axes names and title
    explained_percentage: NDArray[np.float_] = model.explained_var_ratio * 100
    x_title: str = f"{x_name} ({explained_percentage[dim_x]:.1f} % variance)"
    y_title: str = f"{y_name} ({explained_percentage[dim_y]:.1f} % variance)"

    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, c="b")
    plt.title("Projections in the reduced space")
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()
