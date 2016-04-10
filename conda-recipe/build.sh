#!/bin/bash

case `uname` in
    Darwin)
        b2_options=( toolset=clang )
        export MACOSX_DEPLOYMENT_TARGET=10.9
        ;;
    Linux)
        b2_options=( toolset=gcc )
        ;;
esac

$PYTHON setup.py install
