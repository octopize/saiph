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
poetry install
```

If you want to install dev dependencies, make sure you have a rust compiler installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
make install
```

## Documentation

To get the documentation, clone the repo then

```bash
make install docs docs-open
```

## License

Saiph is under MIT license.

## Contributing to Saiph

### Releasing a new version

This whole procedure has to be done on the  `main` branch, as the Github workflow will
automatically create a release when a tag is created, wherever it is.
And unfortunately, we can't combine tags and branch conditions on push
events. More info [here](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#push) and [here](https://stackoverflow.com/questions/57963374/github-actions-tag-filter-with-branch-filter).
```bash
VERSION="0.2.0"

# 1. Edit version in `pyproject.toml` and `saiph/__init__.py`

# 2. Add to next commit
git add pyproject.toml saiph/__init__.py

# 3. Commit
git commit -am "chore: release version $VERSION"

# 4. Tag
git tag $VERSION

# 4. Push
git push && git push --tags
```
