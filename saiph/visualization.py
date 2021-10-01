"""Visualization functions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from saiph.models import Model, Parameters


def plot_circle(
    model: Model, param: Parameters, dimensions=None, min_cor=0.1, max_var=7
):
    """Plot correlation graph.

    Arguments:
        dimensions: dimensions to help by each axis

    Returns:
        plot of the correlation circle
    """
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
    model: Model, param: Parameters, dim=1, max_var=10, min_contrib=0.1
):
    """Plot the variable contribution for a given dimension.

    Args:
        dim: value of the dimension to plot
        max_var: maximum number of variables to plot
        min_contrib: lower threshold for the variable contributions

    Returns:
        graph of the contribution percentages per variables
    """
    # Dimensions start from 1

    # get the useful contributions
    var_contrib = param.contrib[param.contrib.columns[dim - 1]]
    if len(var_contrib) > max_var:
        var_contrib = var_contrib[:max_var]

    # check threshold
    var_contrib = [var for var in var_contrib if var > min_contrib]
    var_contrib = pd.DataFrame(var_contrib)[0]

    indices = list((-var_contrib).argsort())
    names = [list(param.contrib.index)[indices[i]] for i in range(len(indices))]

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(var_contrib)), var_contrib[indices], align="center")
    plt.xticks(range(len(var_contrib)), names, rotation="horizontal")

    # setting labels and title
    plt.title("Variables contributions to Dim. " + str(dim))
    plt.ylabel("Importance")
    plt.xlabel("Variables")
    plt.show()


def plot_explained_var(model: Model, param: Parameters, max_dims=10):
    """Plot explained variance per dimension.

    Args:
        max_dims: maximum number of dimensions to plot

    Return:
        plot of the explained variance
    """
    explained_percentage = model.explained_var_ratio * 100
    if len(explained_percentage) > max_dims:
        explained_percentage = explained_percentage[:max_dims]

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(explained_percentage)), explained_percentage, align="center")
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
