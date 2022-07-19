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
# Perform analyses
# ------------------------
#
model = saiph.stats(model, df)
print(model.cos2)

# %%
# Perform analyses
# ------------------------
#
print(model.contributions)

# %%
# Perform analyses
# ------------------------
#
plot_circle(model=model)
# %%
# Perform analyses
# ------------------------
#
saiph.visualization.plot_explained_var(model)
# %%
# Perform analyses
# ------------------------
#
#saiph.visualization.plot_var_contribution(model.contributions[["Dim. 1","Dim. 2"]], ["param", "2rz"], )