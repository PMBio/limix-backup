/*
 * dist.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */


#include "dist.h"
namespace gpmix{

MatrixXd sq_dist(const MatrixXd x1, const MatrixXd x2)
{

	//1. check everything is aligned
	if (x1.cols()!=x2.cols())
		throw CGPMixException("columns of x1 and x2 not aligned");
	//2. iterate and calculate distances
	MatrixXd RV(x1.rows(),x2.rows());
	for (int i=0;i<x1.rows();i++)
		for (int j=0;j<x2.rows();j++)
		{
			VectorXd d = (x1.row(i)-x2.row(j));
			RV(i,j) = (d.array()*d.array()).sum();
		}
	return RV;
}

}
