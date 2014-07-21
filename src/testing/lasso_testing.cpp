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

#if 0

#include <iostream>
#include "limix/gp/gp_base.h"
#include "limix/types.h"
#include "limix/utils/matrix_helper.h"
#include "limix/likelihood/likelihood.h"
#include "limix/covar/linear.h"
#include "limix/gp/gp_base.h"
#include "limix/lasso/lasso.h"


using namespace std;
using namespace limix;

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!


	MatrixXd X = gpmix::randn((muint_t)100,(muint_t)100);
	MatrixXd y = gpmix::randn((muint_t)100,(muint_t)1);

	MatrixXd w;
	mfloat_t mu = 10;
	lasso_irr(&w,X,y,mu,1E-4,1E-4,10);

	return 0;
}

#endif
