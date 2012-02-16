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
//using namespace std;
//using namespace std::tr1;

#include <string>
#include <tr1/memory>

//define shortcut for shared pointer
#define sptr std::tr1::shared_ptr
#define enable_shared_from_this std::tr1::enable_shared_from_this

namespace gpmix{

#ifndef PI
#define PI 3.14159265358979323846
#endif

const double L2pi = 1.8378770664093453;


//note: for swig it is important that everyhing is typed def and not merely "defined"
typedef double float64_t;
typedef float float32_t;
typedef long int int64_t;
typedef unsigned long int uint64_t;

//default types for usage in GPmix:
typedef float64_t mfloat_t;
typedef int64_t mint_t;
typedef uint64_t muint_t;


//inline casts of exp and log
inline mfloat_t exp (mfloat_t x)
{
		return (mfloat_t)std::exp((long double) x );
}

inline mfloat_t sqrt (mfloat_t x)
{
		return (mfloat_t)std::sqrt((long double) x );
}


inline mfloat_t log (mfloat_t x)
{
		return (mfloat_t)std::log((long double) x );
}

inline mfloat_t inverse (mfloat_t x)
{
	return 1.0/x;
}

//we exclude these int he wrapping section of SWIG
//this is somewhat ugly but makes a lot easier to wrap the Eigen arays with swig
#if (!defined(SWIG) || defined(SWIG_FILE_WITH_INIT))
//standard Matrix type to use in this project
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<mfloat_t, 2, 2,Eigen::ColMajor> MatrixXd2;
typedef Eigen::Matrix<mfloat_t, 3, 3,Eigen::ColMajor> MatrixXd3;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXi;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXi;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXd;
typedef Eigen::Matrix<std::string, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXs;
//typedef Eigen::Array<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> ArrayXd;

//SCIPY matrices for python interface: these are row major
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdscipy;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1> VectorXdscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXiscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1> VectorXiscipy;
#endif

//GpMix Exception Class
class CGPMixException
{
  public:
	CGPMixException(std::string str="Unlabeled exception")
      : What(str)
    {
		std::cout <<"GPMIX Exception: " << str << "\n";
    }


    std::string what()
    {
      return What;
    }

  private:
    std::string What;
};
}


#endif /* TYPES_H_ */
