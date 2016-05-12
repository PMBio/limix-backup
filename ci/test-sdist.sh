#!/bin/bash

set -e -x

FILENAME=`ls limix-sdist/ | head -1`
pip install limix-sdist/$FILENAME
python -c "import limix; print(limix.__version__)"
