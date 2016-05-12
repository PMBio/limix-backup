#!/bin/bash

set -e -x

PYPI_USERNAME=$1
PYPI_PASSWORD=$2

VERSION=`cat limix/setup.py | grep 'VERSION *= *' | sed "s/VERSION *= *'\(.*\)'/\1/g"`

echo $VERSION | grep dev > /dev/null
status_code=$?
if [ status_code -ne 0 ]
then
    exit 0
fi

FILENAME=`ls limix-sdist/ | head -1`
echo $FILENAME
cmd="twine upload -u $PYPI_USERNAME -p $PYPI_PASSWORD limix-sdist/$FILENAME"
for i in {1..5}; do eval $cmd && break || sleep 15; done
