/*
 * types.h
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */

#ifndef TYPES_H_
#define TYPES_H_

#include <Eigen/Dense>
//using namespace Eigen;
#include <string>
using namespace std;
#include <inttypes.h>

namespace gpmix{

//TODO: think whether we really need these?
//some definitions for the python interface
#define float64_t double
#define float32_t float

//default types for usage in GPmix:
#define float_t float64_t
#define int_t int64_t
#define uint_t uint64_t

//inline casts of exp and log
inline float_t exp (float_t x)
{
		return (float_t)std::exp((long double) x );
}

inline float_t log (float_t x)
{
		return (float_t)std::log((long double) x );
}


//standard Matrix type to use in this project
typedef Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<float_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXd;
typedef Eigen::Matrix<string, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXs;
typedef Eigen::Array<float_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> ArrayXd;

//SCIPY matrices for python interface: these are row major
typedef Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdscipy;
typedef Eigen::Matrix<float32_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXfscipy;


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

}


#endif /* TYPES_H_ */
