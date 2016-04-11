Limix developing
================


[![Travis branch](https://img.shields.io/travis/PMBio/limix/master.svg?style=flat-square&label=build%20(osx))](https://travis-ci.org/PMBio/limix) [![Codeship branch](https://img.shields.io/codeship/f2481f10-de6f-0133-67ee-4612197ec823/master.svg?style=flat-square&label=build%20(linux))](https://codeship.com/projects/144764) [![Release](https://img.shields.io/github/release/PMBio/limix.svg?style=flat-square&label=release%20(github))](https://github.com/PMBio/limix/releases/latest) [![PyPI](https://img.shields.io/pypi/v/limix.svg?style=flat-square&label=release%20(pypi))](https://pypi.python.org/pypi/limix/) [![Conda](https://anaconda.org/horta/limix/badges/version.svg)](https://anaconda.org/horta/limix)

## How to modify Limix code?

[![demo](https://asciinema.org/a/dz5tvy04ma8riviemrmg3impm.png)](https://asciinema.org/a/dz5tvy04ma8riviemrmg3impm?autoplay=1)

### Summary

```
git clone https://github.com/PMBio/limix.git
cd limix
git checkout -b my-new-branch
# modify stuff
git add --all
git commit -m "WRITE YOUR COMMIT MESSAGE HERE"
git checkout master
git merge my-new-branch
git push
```

## How to create a Limix release?

[![demo](https://asciinema.org/a/41878.png)](https://asciinema.org/a/41878=1)

### Summary
```
cd limix
git checkout master
bumpversion patch --commit -m "WRITE YOUR RELEASE SUMMARY HERE"
git push
```
