#!/bin/bash

ok=1
base_names=("science-bare" "science-canopy2" "science-conda2")

function abs_script_dir_path
{
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do
      DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
      SOURCE="$(readlink "$SOURCE")"
      [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
}

abs_script_dir_path $0
pushd $DIR >/dev/null

nin ()
{
    local e
    for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
    return 1
}

function help
{
    echo "Usage: $0 [science-bare|science-canopy2|science-conda2]"
}


if [[ $# -ne 1 ]]; then
    help
    ok=0
fi

nin "$1" "${base_names[@]}"
ncontains=$?
if [[ $ok -eq 1 && $ncontains -eq 1 ]]; then
    help
    ok=0
fi

if [[ $ok -eq 1 ]]; then
    BASE_NAME=$1
    docker build -t test.limix.$BASE_NAME -f \
           dockers/test/Dockerfile.$BASE_NAME . \
           && docker run -it test.limix.$BASE_NAME
fi

popd >/dev/null
