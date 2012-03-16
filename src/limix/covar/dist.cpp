/*
 * dist.cpp
 *
 *  Created on: Nov 11, 2011
 *      Author: stegle
 */


#include "dist.h"
namespace limix{

void sq_dist(MatrixXd* out,const MatrixXd& x1, const MatrixXd& x2)
{

	//1. check everything is aligned
	if (x1.cols()!=x2.cols())
		throw CGPMixException("columns of x1 and x2 not aligned");
	//2. iterate and calculate distances
	(*out).resize(x1.rows(),x2.rows());
	for (int i=0;i<x1.rows();i++)
		for (int j=0;j<x2.rows();j++)
		{
			VectorXd d = (x1.row(i)-x2.row(j));
			(*out)(i,j) = (d.array()*d.array()).sum();
		}
}

void lin_dist(MatrixXd* out,const MatrixXd& x1, const MatrixXd& x2,muint_t d)
{
	//1. check everything is aligned
	if (x1.cols()!=x2.cols())
		throw CGPMixException("columns of x1 and x2 not aligned");
	//2. iterate and calculate distances
	(*out).resize(x1.rows(),x2.rows());
	for (int i=0;i<x1.rows();i++)
		for (int j=0;j<x2.rows();j++)
		{
			(*out)(i,j) = x1(i,d)-x2(j,d);
		}
}

}
