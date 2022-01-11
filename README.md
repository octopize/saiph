# Saiph

Saiph not only is the sixth-brightest star in the constellation of Orion (https://en.wikipedia.org/wiki/Saiph), but also a package enabling to project data. 

Projection fitting is done through PCA, MCA or FAMD. 

The main module imputes which one should be used depending on the given data, but each module can be used on his own.

The package provides a visualization module for correlation circles, contributions and explained variance.

See the documentation for more details and a tutorial.

## Documentation

To get the documentation, clone the repo then

```
make install docs docs-open
```

## MacOS M1 prerequisites

install openblas 
```bash
brew install openblas
```

You also probably need to install the package `cython` and `pybind11` (in the dev dependencies) and run the command
```bash
OPENBLAS=$(brew --prefix openblas) CFLAGS="-falign-functions=8 ${CFLAGS}" poetry install
```
source: [github scipy](https://github.com/scipy/scipy/issues/13409)

## License

Saiph is under MIT license.
