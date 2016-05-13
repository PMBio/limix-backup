#!/bin/bash

set -e -x

ls
FILENAME=`ls sdist/ | head -1`
pip install sdist/$FILENAME
python -c "import limix; print(limix.__version__)"
