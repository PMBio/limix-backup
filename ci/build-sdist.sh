#!/bin/bash

set -e -x

pushd limix
  if [ -d "dist" ]
  then
    rm -r dist
  fi
  python setup.py sdist
popd

cp limix/dist/* limix-sdist/
