# Saiph

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Saiph is a Python package for dimensionality reduction and data projection. Named after the sixth-brightest star in the constellation of Orion, Saiph helps you reduce the dimensionality of your datasets using well-established statistical methods.

## Features

- **Automatic Method Selection**: Intelligently chooses the appropriate projection method based on your data types
- **Multiple Projection Methods**:
  - **PCA** (Principal Component Analysis): For continuous numerical data
  - **MCA** (Multiple Correspondence Analysis): For categorical data
  - **FAMD** (Factor Analysis of Mixed Data): For mixed numerical and categorical data
- **Rich Visualizations**: Built-in plotting functions for:
  - Correlation circles
  - Variable contributions
  - Explained variance
  - Individual projections
- **Flexible API**: Use automatic method selection or call specific methods directly
- **Statistical Insights**: Compute cos², contributions, and explained variance for deeper understanding

## Quick Start

### Installation

```bash
pip install saiph
```

### Basic Usage

```python
import pandas as pd
import saiph

# Load your data
df = pd.read_csv("your_data.csv")

# Fit and transform in one step (automatic method selection)
coordinates, model = saiph.fit_transform(df, nf=5)

# Get statistics about the projection
model = saiph.stats(model, df)

# Visualize results
saiph.visualization.plot_projections(model, df, (0, 1))
saiph.visualization.plot_circle(model)
saiph.visualization.plot_explained_var(model)
```

### Using Specific Methods

You can also use PCA, MCA, or FAMD directly:

```python
from saiph.reduction import pca, mca, famd

# For numerical data - PCA
model = pca.fit(df, nf=3)
coordinates = pca.transform(model, df)

# For categorical data - MCA
model = mca.fit(df, nf=3)
coordinates = mca.transform(model, df)

# For mixed data - FAMD
model = famd.fit(df, nf=3)
coordinates = famd.transform(model, df)
```

## API Overview

### Main Functions

- `saiph.fit(df, nf=None, col_weights=None)`: Fit a projection model
- `saiph.transform(model, df)`: Transform data using a fitted model
- `saiph.fit_transform(df, nf=None, col_weights=None)`: Fit and transform in one step
- `saiph.stats(model, df)`: Compute statistical metrics (cos², contributions, explained variance)
- `saiph.inverse_transform(model, coordinates)`: Project back to original space

### Visualization Functions

- `plot_circle(model)`: Display correlation circle
- `plot_var_contribution(values, names)`: Show variable contributions
- `plot_explained_var(model)`: Plot explained variance per dimension
- `plot_projections(model, data, dim)`: Visualize individual projections

## Documentation

For detailed documentation, tutorials, and examples:

```bash
# Clone the repository
git clone https://github.com/octopize/saiph.git
cd saiph

# Install with documentation dependencies
just install

# Build and open documentation
just docs docs-open
```

Or visit the [online documentation](https://saiph.readthedocs.io).

## Advanced Features

### Custom Column Weights

Assign different importance to variables in the projection:

```python
# Give more weight to specific variables
weights = {'age': 2.0, 'income': 1.5}
coordinates, model = saiph.fit_transform(df, nf=3, col_weights=weights)
```

### Sparse Data Support

For large datasets with many categorical variables:

```python
# Use sparse FAMD for better memory efficiency
coordinates, model = saiph.fit_transform(df, nf=5, sparse=True)
```

### Inverse Transform

Project coordinates back to the original space:

```python
# Get reduced coordinates
coordinates, model = saiph.fit_transform(df, nf=3)

# Project back
reconstructed_df = saiph.inverse_transform(model, coordinates)
```

## Examples

### Example 1: Iris Dataset with PCA

```python
import pandas as pd
import saiph

# Load iris dataset
df = pd.read_csv("iris.csv")

# Automatic projection (will use PCA for numerical data)
coords, model = saiph.fit_transform(df.drop('species', axis=1), nf=2)

# Compute statistics
model = saiph.stats(model, df.drop('species', axis=1))

# Visualize
import matplotlib.pyplot as plt
saiph.visualization.plot_projections(model, df.drop('species', axis=1), (0, 1))
saiph.visualization.plot_circle(model)
plt.show()
```

### Example 2: Mixed Data with FAMD

```python
import pandas as pd
import saiph

# Load dataset with mixed types (numerical + categorical)
df = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 75000, 80000],
    'education': ['high_school', 'bachelor', 'master', 'phd'],
    'city': ['NYC', 'LA', 'NYC', 'SF']
})

# FAMD automatically selected for mixed data
coords, model = saiph.fit_transform(df, nf=3)
model = saiph.stats(model, df)

# Check explained variance
print(f"Explained variance ratio: {model.explained_var_ratio}")

# Visualize contributions
saiph.visualization.plot_var_contribution(
    model.contributions["Dim. 1"].to_numpy(),
    model.contributions.index.to_numpy()
)
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --extra matplotlib --group dev --group doc
```

If you want to install dev dependencies, make sure you have a rust compiler installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
just install
```

### Running Tests

```bash
# Run all tests
just test

# Run specific tests
uv run pytest saiph/projection_test.py
```

### Code Quality

```bash
# Run linting
just lint

# Auto-fix linting issues
just lint-fix

# Type checking
just typecheck

# Run all checks
just ci
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

### Releasing

To release a new version:

```bash
uv run python release.py --bump-type {patch, minor, major}
```

## License

Saiph is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use Saiph in your research, please cite:

```bibtex
@software{saiph,
  title = {Saiph: A Python Package for Dimensionality Reduction},
  author = {Octopize},
  year = {2024},
  url = {https://github.com/octopize/saiph}
}
```

## Support

- 📧 Email: help@octopize.io
- 🐛 Issues: [GitHub Issues](https://github.com/octopize/saiph/issues)
- 📖 Documentation: [Read the Docs](https://saiph.readthedocs.io)