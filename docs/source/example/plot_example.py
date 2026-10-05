"""
Complete Saiph Tutorial
=========================

This example demonstrates a complete workflow using Saiph for dimensionality reduction.

We'll cover:

1. Loading and preparing data
2. Fitting a projection model (automatic method selection)
3. Transforming data to reduced dimensions
4. Computing statistical metrics
5. Visualizing results with various plots

This tutorial uses the Iris dataset as an example.
"""

import pandas as pd

import saiph
from saiph.visualization import plot_circle

# %%
# Step 1: Load the Data
# ------------------------
# First, we load a sample dataset. Saiph will automatically detect
# the data types and choose the appropriate projection method.
#
# The Iris dataset contains 4 numerical features (sepal length, sepal width,
# petal length, petal width), making it suitable for PCA.

df = pd.read_csv("../../../tests/fixtures/iris.csv")
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# %%
# Step 2: Fit and Transform the Model
# ------------------------
# The `fit_transform()` function will:
# - Automatically select PCA (since all features are numerical)
# - Fit the model on the data
# - Transform the data to the reduced space
#
# We specify `nf=5` to keep 5 principal components, but you can adjust
# this based on your needs or the explained variance.

coord, model = saiph.fit_transform(df, nf=5)
print("\nProjected coordinates shape:", coord.shape)
print("\nFirst few projected coordinates:")
print(coord.head())

# %%
# Step 3: Visualize Individual Projections
# ------------------------
# Plot the individuals in the reduced 2D space (first two dimensions).
# Each point represents an observation from the original dataset.
#
saiph.visualization.plot_projections(model, df, (0, 1))

# %%
# Step 4: Compute Statistical Metrics
# ------------------------
# The `stats()` function computes important metrics:
# - cos²: Quality of representation for each variable
# - contributions: How much each variable contributes to each dimension
# - explained_var: Variance explained by each dimension
# - explained_var_ratio: Proportion of total variance explained
#
model = saiph.stats(model, df)
print("\nCos² (quality of representation):")
print(model.cos2)

# %%
# Step 5: Correlation Circle
# ------------------------
# The correlation circle shows how original variables are correlated
# with the principal components. Variables pointing in similar directions
# are positively correlated.
#
plot_circle(model=model)

# %%
# Step 6: Variable Contributions
# ------------------------
# Variable contributions show which original variables contribute most
# to each principal component. Higher values indicate more importance.
#
print("\nVariable contributions to each dimension:")
print(model.contributions)

# %%
# Visualize Contributions for Dimension 1
# ------------------------
# This bar plot shows the contribution of each variable to the first
# principal component.
#
saiph.visualization.plot_var_contribution(
    model.contributions["Dim. 1"].to_numpy(), model.contributions.index.to_numpy()
)


# %%
# Step 7: Explained Variance
# ------------------------
# Understanding how much variance each component explains helps you
# decide how many components to keep.
#
print("\nExplained variance by dimension:")
print(model.explained_var)

# %%
# Explained Variance Ratio
# ------------------------
# This shows the proportion of total variance explained by each component.
# Typically, you want to keep enough components to explain 70-90% of variance.
#
print("\nExplained variance ratio:")
print(model.explained_var_ratio)
print(f"\nCumulative explained variance: {model.explained_var_ratio.sum():.2%}")

# %%
# Visualize Explained Variance
# ------------------------
# A bar plot makes it easy to see which components explain the most variance.
#
saiph.visualization.plot_explained_var(model)

# %%
# Step 8: Transform New Data
# ------------------------
# Once you have a fitted model, you can transform new observations
# into the same reduced space.
#
# Here we demonstrate by transforming the first 10 rows:
new_data = df.head(10)
new_coords = saiph.transform(new_data, model)
print("\nNew data projected coordinates:")
print(new_coords)

# %%
# Conclusion
# ------------------------
# This tutorial demonstrated the complete Saiph workflow:
#
# - Automatic method selection based on data types
# - Model fitting and data transformation
# - Statistical analysis (cos², contributions, explained variance)
# - Visualization (projections, correlation circle, contributions, variance)
# - Transforming new data using a fitted model
#
# For mixed data types, Saiph will automatically use FAMD instead of PCA.
# You can also use specific methods directly: `saiph.reduction.pca`,
# `saiph.reduction.mca`, or `saiph.reduction.famd`.
