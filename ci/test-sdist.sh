#!/bin/bash

set -e -x

FILENAME=`ls sdist/ | head -1`
pip install sdist/$FILENAME
python -c "import limix; print(limix.__version__)"
