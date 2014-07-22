// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.

#include "dist.h"
namespace limix{

void sq_dist(MatrixXd* out,const MatrixXd& x1, const MatrixXd& x2)
{

	//1. check everything is aligned
	if (x1.cols()!=x2.cols())
		throw CLimixException("columns of x1 and x2 not aligned");
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
		throw CLimixException("columns of x1 and x2 not aligned");
	//2. iterate and calculate distances
	(*out).resize(x1.rows(),x2.rows());
	for (int i=0;i<x1.rows();i++)
		for (int j=0;j<x2.rows();j++)
		{
			(*out)(i,j) = x1(i,d)-x2(j,d);
		}
}

}
