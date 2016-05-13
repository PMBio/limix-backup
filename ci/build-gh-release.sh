#!/bin/bash

set -e -x

is=`limix/ci/is_version_releasable.sh`
if [ "$is" == "false" ]
then
    exit 0
fi

VERSION=`limix/ci/get_version.sh`

cp dist/* limix-gh-release/
echo "Limix $VERSION" > limix-gh-release/name
echo "v$VERSION" > limix-gh-release/tag

pushd limix
  commit_msg=`git log -1 --pretty=%B | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | grep -v '^$'`
popd

fullmsg="This is an automatically generated release triggered by a commit with the following message: $commit_msg"
echo $fullmsg > limix-gh-release/body
