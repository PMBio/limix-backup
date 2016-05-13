#!/bin/bash

set -e -x

is=`limix/ci/is_version_releasable.sh`

if [ "$is" == "true" ]
then
    FILENAME=`ls limix-sdist/ | head -1`
    twine upload -u $PYPI_USERNAME -p $PYPI_PASSWORD limix-sdist/$FILENAME
fi
