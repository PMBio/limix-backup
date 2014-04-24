/* -*- C -*-  (not really, but good for syntax highlighting) */


%apply float { mfloat_t };

%typecheck(SWIG_TYPECHECK_INTEGER)
	   int, short, long,
 	   unsigned int, unsigned short, unsigned long,
	   signed char, unsigned char,
	   long long, unsigned long long,
	   const int &, const short &, const long &,
 	   const unsigned int &, const unsigned short &, const unsigned long &,
	   const long long &, const unsigned long long &,
	   enum SWIGTYPE,
           bool, const bool & 
{
  $1 = (PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;
}


%typecheck(SWIG_TYPECHECK_FLOAT) 
           mfloat_t,
           const mfloat_t,
           mfloat_t & {

  $1 = (PyFloat_Check($input) || PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;

}

%typecheck(SWIG_TYPECHECK_FLOAT_ARRAY) 
         MatrixXd, 
         MatrixXd *,
         const MatrixXd,
         MatrixXd &,
         const MatrixXd & {
  $1 = (array_type($input) == PyArray_FLOAT) || (array_type($input) == PyArray_DOUBLE);
}



//MatrixXd
%typemap(in,
         fragment="NumPy_Fragments") 
         const MatrixXd & (MatrixXd temp) {

    // create array from input
    int newObject=0;
    PyArrayObject * in_array;
    
    switch ( array_type($input) ) {

    case PyArray_LONG:
    case PyArray_DOUBLE:
      in_array = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE, &newObject);
      break;

    case PyArray_INT:
    case PyArray_FLOAT:

    default:
      PyErr_SetString(PyExc_ValueError,
                      "array must be of type int, float, long or double");
      
      return NULL;
    }

    if( in_array == NULL ){
      PyErr_SetString(PyExc_ValueError,
                      "array could not be created");
      
      return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);

    // get the dimensions
    int in_rows;
    int in_cols;
    if(array_numdims(in_array) == 2){

      in_rows = array_size(in_array, 0);
      in_cols = array_size(in_array, 1);

    }else{
	  //if vector: create a column vector explicitly:
	  in_rows = array_size(in_array, 0);
	  in_cols = 1;
    }

    $1 = &temp;
    // prepare the input array
    switch( array_type($input) ) {

    case PyArray_LONG:
    case PyArray_DOUBLE:
      (*$1) = Eigen::Map<MatrixXdscipy>((double*)array_data( in_array ), in_rows, in_cols).cast<mfloat_t>();
      break;
    case PyArray_INT:
    case PyArray_FLOAT:

    default:
      PyErr_SetString(PyExc_ValueError,
                      "array must be of type int, float, long or double");
      return NULL;
    }
    //refernce counter if we craeted a copy?
    if(newObject)
      {
	Py_DECREF(in_array);
      }

}

%typemap(in, numinputs = 0) 
MatrixXd* (MatrixXd temp) {
	
	$1 = &temp;
	
}

%typemap(argout) 
         MatrixXd* {

  // prepare resulting array
  npy_intp dims[] = {$1->rows(), $1->cols()};
  PyObject * out_array = PyArray_SimpleNew(2, dims, PyArray_DOUBLE);

  if (out_array == NULL){
    PyErr_SetString(PyExc_ValueError,
                    "Unable to create the output array.");
    
    return NULL;
  }
  
  mfloat_t* out_data = (mfloat_t*)array_data(out_array);
  Eigen::Map<MatrixXdscipy>(out_data, dims[0], dims[1]) = (*$1);

  $result = SWIG_Python_AppendOutput($result, out_array);
}
//end MatrixXd



//MatrixXi
%typemap(in,
         fragment="NumPy_Fragments") 
const MatrixXi & (MatrixXi temp) {
	
    // create array from input
    int newObject=0;
    PyArrayObject * in_array;
    
    switch ( array_type($input) ) {
			
		case PyArray_LONG:
		case PyArray_DOUBLE:
			in_array = obj_to_array_contiguous_allow_conversion($input, PyArray_INT64, &newObject);
			break;
		case PyArray_INT:
			in_array = obj_to_array_contiguous_allow_conversion($input, PyArray_INT64, &newObject);
			break;
		case PyArray_FLOAT:
			
		default:
			PyErr_SetString(PyExc_ValueError,
							"array must be of type int, float, long or double");
			
			return NULL;
    }
	
    if( in_array == NULL ){
		PyErr_SetString(PyExc_ValueError,
						"array could not be created");
		
		return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);
	
    // get the dimensions
    int in_rows;
    int in_cols;
    if(array_numdims(in_array) == 2){
		
		in_rows = array_size(in_array, 0);
		in_cols = array_size(in_array, 1);
		
    }else{
		//if vector: create a column vector explicitly:
		in_rows = array_size(in_array, 0);
		in_cols = 1;
    }
	
    $1 = &temp;
    // prepare the input array
    switch( array_type($input) ) {
			
		case PyArray_LONG:
		case PyArray_DOUBLE:
		case PyArray_INT:
			(*$1) = Eigen::Map<MatrixXiscipy>((mint_t*)array_data( in_array ), in_rows, in_cols).cast<mint_t>();
			break;
		case PyArray_FLOAT:
			
		default:
			PyErr_SetString(PyExc_ValueError,
							"array must be of type int, float, long or double");
			return NULL;
    }
    //refernce counter if we craeted a copy?
    if(newObject)
      {
	Py_DECREF(in_array);
      }
}



%typemap(in, numinputs = 0) 
MatrixXi* (MatrixXi temp) {
	$1 = &temp;
}


%typemap(argout) 
MatrixXi* {
	
	// prepare resulting array
	npy_intp dims[] = {$1->rows(), $1->cols()};
	PyObject * out_array = PyArray_SimpleNew(2, dims, PyArray_INT64);
	
	if (out_array == NULL){
		PyErr_SetString(PyExc_ValueError,
						"Unable to create the output array.");
		
		return NULL;
	}
	
	mint_t* out_data = (mint_t*)array_data(out_array);
	Eigen::Map<MatrixXiscipy>(out_data, dims[0], dims[1]) = (*$1);
	
	$result = SWIG_Python_AppendOutput($result, out_array);
}
//end MatrixXi






/*VectorXd*/
%typemap(in,
         fragment="NumPy_Fragments") 
         const VectorXd & (VectorXd temp) {

    // create array from input
    int newObject=0;
    PyArrayObject * in_array;
    
    switch ( array_type($input) ) {

    case PyArray_LONG:
    case PyArray_DOUBLE:
      in_array = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE, &newObject);
      break;

    case PyArray_INT:
    case PyArray_FLOAT:

    default:
      PyErr_SetString(PyExc_ValueError,
                      "array must be of type int, float, long or double");
      
      return NULL;
    }

    if( in_array == NULL ){
      PyErr_SetString(PyExc_ValueError,
                      "array could not be created");
      
      return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);

    // get the dimensions
    int in_rows;
    int in_cols;
    if(array_numdims(in_array) == 2){

      in_rows = array_size(in_array, 0);
      in_cols = array_size(in_array, 1);

    }else{

      in_rows = array_size(in_array, 0);
      in_cols = 1;

    }

    $1 = &temp;

    // prepare the input array
    switch( array_type($input) ) {

    case PyArray_LONG:
    case PyArray_DOUBLE:
      (*$1) = Eigen::Map<VectorXdscipy>((double*)array_data( in_array ), in_rows).cast<mfloat_t>();
      break;
    case PyArray_INT:
    case PyArray_FLOAT:

    default:
      PyErr_SetString(PyExc_ValueError,
                      "array must be of type int, float, long or double");
      return NULL;
    }
    //refernce counter if we craeted a copy?
    if(newObject)
      {
	Py_DECREF(in_array);
      }
}




%typemap(in, numinputs = 0) 
         VectorXd* (VectorXd temp) {

  $1 = &temp;

}

%typemap(argout) 
         VectorXd* {

  // prepare resulting array
  /*
  npy_intp dims[] = {$1->rows(), $1->cols()};
  PyObject * out_array = PyArray_SimpleNew(2, dims, PyArray_DOUBLE);

  if (out_array == NULL){
    PyErr_SetString(PyExc_ValueError,
                    "Unable to create the output array.");
    
    return NULL;
  }
  
  mfloat_t* out_data = (mfloat_t*)array_data(out_array);
  Eigen::Map<VectorXdscipy>(out_data, dims[0]) = (*$1);

  $result = SWIG_Python_AppendOutput($result, out_array);
  */
  //Vector types in eigen have rows only:
  npy_intp dims[] = {$1->rows()};
  PyObject * out_array = PyArray_SimpleNew(1, dims, PyArray_DOUBLE);

  if (out_array == NULL){
    PyErr_SetString(PyExc_ValueError,
                    "Unable to create the output array.");
    
    return NULL;
  }
  
  mfloat_t* out_data = (mfloat_t*)array_data(out_array);
  Eigen::Map<VectorXdscipy>(out_data, dims[0]) = (*$1);

  $result = SWIG_Python_AppendOutput($result, out_array);
}


//end VectorXd




/*VectorXi*/
%typemap(in,
         fragment="NumPy_Fragments") 
const VectorXi& (VectorXi temp) {
	
    // create array from input
    int newObject=0;
    PyArrayObject * in_array;
    
    switch ( array_type($input) ) {
			
		case PyArray_LONG:
		case PyArray_DOUBLE:
		case PyArray_INT:
			in_array = obj_to_array_contiguous_allow_conversion($input, PyArray_INT64, &newObject);
			break;
		case PyArray_FLOAT:
			
		default:
			PyErr_SetString(PyExc_ValueError,
							"array must be of type int, float, long or double");
			
			return NULL;
    }
	
    if( in_array == NULL ){
		PyErr_SetString(PyExc_ValueError,
						"array could not be created");
		
		return NULL;
    }
    
    // require one or two dimensions
    int dims[] = {1, 2};
    require_dimensions_n(in_array, dims, 2);
	
    // get the dimensions
    int in_rows;
    int in_cols;
    if(array_numdims(in_array) == 2){
		
		in_rows = array_size(in_array, 0);
		in_cols = array_size(in_array, 1);
		
    }else{
		
		in_rows = array_size(in_array, 0);
		in_cols = 1;
		
    }
	
    $1 = &temp;
	
    // prepare the input array
    switch( array_type($input) ) {
			
		case PyArray_LONG:
		case PyArray_DOUBLE:
		case PyArray_INT:
			(*$1) = Eigen::Map<VectorXiscipy>((mint_t*)array_data( in_array ), in_rows)	.cast<mint_t>();
			break;
		case PyArray_FLOAT:
			
		default:
			PyErr_SetString(PyExc_ValueError,
							"array must be of type int, float, long or double");
			return NULL;
    }
    //refernce counter if we craeted a copy?
    if(newObject)
      {
	Py_DECREF(in_array);
      }

}




%typemap(in, numinputs = 0) 
VectorXi* (VectorXi temp) {
	
	$1 = &temp;
	
}

%typemap(argout) 
VectorXi* {
	//Vector types in eigen have rows only:
	npy_intp dims[] = {$1->rows()};
	PyObject * out_array = PyArray_SimpleNew(1, dims, PyArray_INT64);
	
	if (out_array == NULL){
		PyErr_SetString(PyExc_ValueError,
						"Unable to create the output array.");
		
		return NULL;
	}
	
	mint_t* out_data = (mint_t*)array_data(out_array);
	Eigen::Map<VectorXiscipy>(out_data, dims[0]) = (*$1);
	
	$result = SWIG_Python_AppendOutput($result, out_array);
}
//end VectorXi







%typemap(argout) 
         mfloat_t* {

  $result = SWIG_Python_AppendOutput($result, Py_BuildValue("f", $1));
}


%typemap(in, numinputs = 0) 
         int* (int temp) {

  $1 = &temp;

}

%typemap(argout) 
         int* {

  $result = SWIG_Python_AppendOutput($result, Py_BuildValue("i", $1));
}
