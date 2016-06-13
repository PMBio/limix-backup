Limix developing
================

## Modifying Limix

[![demo](https://asciinema.org/a/dz5tvy04ma8riviemrmg3impm.png)](https://asciinema.org/a/dz5tvy04ma8riviemrmg3impm?autoplay=1)

### Summary

```
git clone https://github.com/PMBio/limix.git
cd limix
git checkout -b my-new-branch
# modify stuff
git add --all
git commit -m "WRITE YOUR COMMIT MESSAGE HERE"
git checkout master
git merge my-new-branch
git push
```

## Releasing Limix

[![demo](https://asciinema.org/a/41878.png)](https://asciinema.org/a/41878=1)

### Summary
```
cd limix
git checkout master
bumpversion patch --commit -m "WRITE YOUR RELEASE SUMMARY HERE"
git push
```

## Limix status

- OSX tests and Conda OSX build are performed using Travis:
https://travis-ci.org/PMBio/limix.
- Linux tests, GitHub TAG and RELEASE, PyPI deployment, and Conda Linux build
are performed using Codeship: https://codeship.com/projects/144764.
- Windows tests and Conda Windows build are performed using AppVeyor: https://ci.appveyor.com/project/Horta/limix. (This is not working yet.)
