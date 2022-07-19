"""
Tutorial
=========================

This example show how saiph works.
"""

import saiph
import pandas as pd

from saiph.visualization import plot_circle 


# %%
# Fit the model
# ------------------------

df = pd.read_csv("../../tests/fixtures/iris.csv")

coord, model = saiph.fit_transform(df, nf=5)
print(coord.head())
# %%
# Individuals projection
# ------------------------
#
saiph.visualization.plot_projections(model, df, (0, 1))
# %%
# Perform statistics on the projection
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
saiph.visualization.plot_var_contribution(model.contributions["Dim. 1"], model.contributions.index )


# %%
# Explained variance
# ------------------------
#
saiph.visualization.plot_explained_var(model)

