#!/bin/bash

python setup.py bdist_wheel
virtualenv --python=python venv-for-wheel
. venv-for-wheel/bin/activate
pushd dist
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade wheel
pip install --pre --upgrade --find-links . limix
popd
