#!/bin/bash

VERSION=`cat limix/setup.py | grep 'VERSION *= *' | sed "s/VERSION *= *'\(.*\)'/\1/g"`
echo $VERSION
