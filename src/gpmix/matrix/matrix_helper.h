/*
 * matrix_helper.h
 *
 *  Created on: Nov 10, 2011
 *      Author: stegle
 */

#ifndef MATRIX_HELPER_H_
#define MATRIX_HELPER_H_

#include <math.h>
#include <cmath>
#include <string>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

//create random matrix:
double randn(double mu=0.0, double sigma=1.0);


//TODO: think whether we really need these?
//some definitions for the python interface
#define float64_t double
#define float32_t float
#define int32_t int

//standard Matrix type to use in this project
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
typedef Matrix<double, Dynamic, 1> VectorXd;
typedef Matrix<string, Dynamic, 1> VectorXs;


//helper functions for eigen matrices
bool isnull(const MatrixXd& m);
double sum(MatrixXd& m);
MatrixXd log(MatrixXd& m);



MatrixXd randn(const unsigned int n, const unsigned int m);
MatrixXd rand(const unsigned int n,const unsigned int m);


#endif /* MATRIX_HELPER_H_ */
