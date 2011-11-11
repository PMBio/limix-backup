/*
 * types.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Dense>
using namespace Eigen;
#include <string>
using namespace std;
#include <inttypes.h>



//TODO: think whether we really need these?
//some definitions for the python interface
#define float64_t double
#define float32_t float

//default types for usage in GPmix:
#define float_t float64_t
#define int_t int64_t
#define uint_t uint64_t


//standard Matrix type to use in this project
typedef Matrix<float_t, Dynamic, Dynamic> MatrixXd;
typedef Matrix<float_t, Dynamic, 1> VectorXd;
typedef Matrix<string, Dynamic, 1> VectorXs;
typedef Array<float_t, Dynamic, Dynamic> ArrayXd;


class CGPMixException
{
  public:

	CGPMixException()
	      : What("Unlabeled Exception")
	    {
	    }

	CGPMixException(string str)
      : What(str)
    {
    }

    string what()
    {
      return What;
    }

  private:
    string What;
};




#endif /* TYPES_H_ */
