[![Travis branch](https://img.shields.io/travis/PMBio/limix/master.svg?style=flat-square&label=build%20(osx))](https://travis-ci.org/PMBio/limix) [![Codeship branch](https://img.shields.io/codeship/f2481f10-de6f-0133-67ee-4612197ec823/master.svg?style=flat-square&label=build%20(linux))](https://codeship.com/projects/144764) [![Release](https://img.shields.io/github/release/PMBio/limix.svg?style=flat-square&label=release%20(github))](https://github.com/PMBio/limix/releases/latest) [![PyPI](https://img.shields.io/pypi/v/limix.svg?style=flat-square&label=release%20(pypi))](https://pypi.python.org/pypi/limix/) [![Conda](https://anaconda.org/horta/limix/badges/version.svg)](https://anaconda.org/horta/limix) [![License](http://img.shields.io/:license-apache-blue.svg?style=flat-square)](https://github.com/PMBio/limix/blob/master/LICENSE.txt)

Limix
=====

Limix is a flexible and efficient linear mixed model library with interfaces
to Python. Genomic analyses require flexible models that can be adapted to the needs of
the user. Limix is smart about how particular models are fitted to save
computational cost.

## Installation

### Using Conda package manager

Conda is a package manager designed for Python and R users/developers of
scientific tools, and comes with the [Anaconda distribution](https://www.continuum.io/downloads).
Currently we support this installation for Linux 64 bits and OSX operating
systems.

```
conda install -c https://conda.anaconda.org/horta limix
```

### Using Pip

If you don't have Conda (or don't want to use the above method), Limix can be
installed via Pip package manager.
```
pip install limix
```
This approach is not as straightforward as the first one because it requires
compilation of C/C++ and (potentially) Fortran code, and some understanding
of dependency resolution is likely to be required. We provide bellow recipes
for some popular Limix distributions, assuming you have the `wget` command line
tool.

- Ubuntu

    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/apt_limix_install)
    ```

- Fedora
    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/dnf_limix_install)
    ```

- OpenSUSE
    ```
    bash <(wget -O - https://raw.githubusercontent.com/PMBio/limix/master/deploy/zypper_limix_install)
    ```

### From source

This is more tricky in terms of dependency resolution but useful for developers.

```
git clone https://github.com/PMBio/limix.git
cd limix
python setup.py install # or python setup.py develop
```

## Usage

A good starting point is our package Vignettes. These tutorials are available from this repository: https://github.com/PMBio/limix-tutorials.

The main package vignette can also be viewed using the ipython notebook viewer:
http://nbviewer.ipython.org/github/pmbio/limix-tutorials/blob/master/index.ipynb.

Alternatively, the source file is available in the separate Limix tutorial repository:
https://github.com/PMBio/limix-tutorials

## Problems

If you want to use Limix and encounter any issues, please contact us via `limix@mixed-models.org`.

## Authors

- `Franceso Paolo Casale` (`casale@ebi.ac.uk`)
- `Danilo Horta` (`horta@ebi.ac.uk`)
- `Christoph Lippert` (`christoph.a.lippert@gmail.com`)
- `Oliver Stegle` (`stegle@ebi.ac.uk`)

## License

See [Apache License (Version 2.0, January 2004)](https://github.com/PMBio/limix/blob/master/LICENSE.txt).
