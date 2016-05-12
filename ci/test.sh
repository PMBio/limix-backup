#!/bin/bash

set -e -x

pushd limix
  find . -type f -name "*.so" -delete
  python setup.py test --yes
popd
