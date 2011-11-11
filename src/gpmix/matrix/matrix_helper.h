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
#include <gpmix/types.h>


//create random matrix:
double randn(double mu=0.0, double sigma=1.0);
//helper functions for eigen matrices
bool isnull(const MatrixXd& m);
double sum(MatrixXd& m);
MatrixXd log(MatrixXd& m);



MatrixXd randn(const uint_t n, const uint_t m);
MatrixXd rand(const uint_t n,const uint_t m);


#endif /* MATRIX_HELPER_H_ */
