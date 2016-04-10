#!/bin/bash

case `uname` in
    Darwin)
        export MACOSX_DEPLOYMENT_TARGET=10.9
        ;;
    Linux)
        ;;
esac

$PYTHON setup.py install --compatible
