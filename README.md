# Saiph

Saiph not only is the sixth-brightest star in the constellation of Orion (https://en.wikipedia.org/wiki/Saiph), but also a package enabling to project data. 

Projection fitting is done through PCA, MCA or FAMD. 

The main module imputes which one should be used depending on the given data, but each module can be used on his own.

The package provides a visualization module for correlation circles, contributions and explained variance.

See the documentation for more details and a tutorial.

## Install

```bash
pip install saiph
```

## Development

```bash
uv sync --extra matplotlib --extra dev
```

If you want to install dev dependencies, make sure you have a rust compiler installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
just install
```

## Documentation

To get the documentation, clone the repo then

```bash
just install docs docs-open
```

## License

Saiph is under MIT license.