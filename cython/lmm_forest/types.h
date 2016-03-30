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


#ifndef PI
#define PI 3.14159265358979323846
#endif

const double L2pi = 1.8378770664093453;


//note: for swig it is important that everyhing is typed def and not merely "defined"
typedef double float64_t;
typedef float float32_t;
//typedef long int int64_t;
//typedef unsigned long int uint64_t;

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

typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXd;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdRM;
typedef Eigen::Matrix<mfloat_t, 2, 2,Eigen::ColMajor> MatrixXd2;
typedef Eigen::Matrix<mfloat_t, 3, 3,Eigen::ColMajor> MatrixXd3;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::ColMajor> MatrixXi;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXi;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXd;
typedef Eigen::Matrix<string, Eigen::Dynamic, 1,Eigen::ColMajor> VectorXs;
//typedef Eigen::Array<mint_t, Eigen::Dynamic, 1,Eigen::ColMajor> ArrayXi;

//maps for python
typedef Eigen::Map<MatrixXi> MMatrixXi;
typedef Eigen::Map<MatrixXd> MMatrixXd;
typedef Eigen::Map<MatrixXdRM> MMatrixXdRM;

typedef Eigen::Map<VectorXd> MVectorXd;



//SCIPY matrices for python interface: these are row major
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXdscipy;
typedef Eigen::Matrix<mfloat_t, Eigen::Dynamic, 1> VectorXdscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, Eigen::Dynamic,Eigen::RowMajor> MatrixXiscipy;
typedef Eigen::Matrix<mint_t, Eigen::Dynamic, 1> VectorXiscipy;






#endif /* TYPES_H_ */
