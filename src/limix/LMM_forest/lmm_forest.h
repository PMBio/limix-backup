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

#ifndef LMM_FOREST_H_
#define LMM_FOREST_H_

#include "SplittingCore.h"


namespace limix {

	//TODO fill me with function you want to call from python
	void FitMyTreeCrapStuff(MatrixXd* matrix_out,const MatrixXd& matrix_in);
	//from python matrix_out = limix.FitMyTreeCrapStuff(matrix_in)
	// A
	//
	void best_split_full_model(mint_t* m_best, 
                               mfloat_t* s_best,
                               mfloat_t* left_mean,
		                       mfloat_t* right_mean,
		                       mfloat_t* ll_score,
		                       const MatrixXd& X,
		                       const MatrixXd& UTy,
		                       const MatrixXd& C,
		                       const MatrixXd& S,
		                       const MatrixXd& U,
		                       const VectorXi& noderange,
		                       mfloat_t delta);
}
#endif
