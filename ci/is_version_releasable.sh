#!/bin/bash

VERSION=`limix/ci/get_version.sh`

echo $VERSION | grep -E "^\d+\.\d+\.\d+(\.(a|b|rc)\d+)?$" > /dev/null

status_code=$?
if [ "$status_code" -ne 0 ]
then
    echo "false"
else
    echo "true"
fi
