"""
Tutorial
=========================

This example show how saiph works.
"""

import pandas as pd

import saiph
from saiph.visualization import plot_circle

# %%
# Fit the model
# ------------------------

df = pd.read_csv("../../tests/fixtures/iris.csv")

coord, model = saiph.fit_transform(df, nf=5)
print(coord.head())
# %%
# Project individuals
# ------------------------
#
saiph.visualization.plot_projections(model, df, (0, 1))
# %%
# Get statistics about the projection
# ------------------------
#
model = saiph.stats(model, df)
print(model.cos2)

# %%
# Circle of correlations
# ------------------------
#
plot_circle(model=model)

# %%
# Variable contributions
# ------------------------
#
print(model.contributions)
#%%
# ------------------------
#
saiph.visualization.plot_var_contribution(
    model.contributions["Dim. 1"].to_numpy(), model.contributions.index.to_numpy()
)


# %%
# Explained variance
# ------------------------
#
saiph.visualization.plot_explained_var(model)
