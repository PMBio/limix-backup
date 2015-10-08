# LIMIX

## What is LIMIX?

LIMIX is a flexible and efficient linear mixed model library with interfaces to Python. 
Limix is currently mainly developed by 

Franceso Paolo Casale (casale@ebi.ac.uk)
Danilo Horta (horta@ebi.ac.uk) 
Christoph Lippert (lippert@microsoft.com) 
Oliver Stegle (stegle@ebi.ac.uk) 


## Philosophy 

Genomic analyses require flexible models that can be adapted to the needs of the user. 
LIMIX is smart about how particular models are fitted to save computational cost. 


## Installation:

* It is recommended to install LIMIX via pypi.
* pip install limix will work on most systems.
* LIMIX is especially easy to install with the anaconda python distribution: https://store.continuum.io/cshop/anaconda.

Here are the requirementss to want to install LIMIX from source:

* Python:
  * scipy, numpy, pandas, cython

* Swig:
  * swig 2.0 or higher (only required if you need to recompile C++ interfaces)

## How to use LIMIX?

A good starting point is our package Vignettes. These tutorials are available from this repository: https://github.com/PMBio/limix-tutorials.

The main package vignette can also be viewed using the ipython notebook viewer:
http://nbviewer.ipython.org/github/pmbio/limix-tutorials/blob/master/index.ipynb.

Alternatively, the source file is available in the separate LIMIX tutorial repository:
https://github.com/PMBio/limix-tutorials

## Problems ? 
If you want to use LIMIX and encounter any issues, please contact us by email: limix@mixed-models.org

## License
See [LICENSE] https://github.com/PMBio/limix/blob/master/license.txt
