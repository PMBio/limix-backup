#!/bin/bash

#make sure '.' is in the pathy such that py_filter is found
export PATH=$PATH:.

#call doxygen
doxygen ./doxy.cfg
