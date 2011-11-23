%module gpmix

%{
#define SWIG_FILE_WITH_INIT
#define SWIG
#include "gpmix/types.h"
#include "gpmix/LMM/lmm.h"
  using namespace gpmix;
%}


/* Get the numpy typemaps */
%include "numpy.i"
%include "typemaps.i"



%init %{
  import_array();
%}




%apply (short*  IN_ARRAY1, int DIM1) {(short*  series, int size)};
%apply (int*    IN_ARRAY1, int DIM1) {(int*    series, int size)};
%apply (long*   IN_ARRAY1, int DIM1) {(long*   series, int size)};
%apply (float*  IN_ARRAY1, int DIM1) {(float*  series, int size)};
%apply (double* IN_ARRAY1, int DIM1) {(double* series, int size)};

%apply (int*    IN_ARRAY2, int DIM1, int DIM2) {(int*    matrix, int rows, int cols)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* matrix, int rows, int cols)};

%apply (int*    INPLACE_ARRAY1, int DIM1) {(int*    array,   int size)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* array,   int size)};



/* Remove C Prefix 
%rename(VBFA) cVBFA;
%rename(PEER) cSPARSEFA;
*/

/* Include the header file to be wrapped */
%include "gpmix/LMM/lmm.h"


 
