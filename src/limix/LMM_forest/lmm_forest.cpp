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

#include "lmm_forest.h"

namespace limix {

	void FitMyTreeCrapStuff(MatrixXd* matrix_out, const MatrixXd& matrix_in)
	{
    }

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
		                       mfloat_t delta){

       C_best_split_full_model(m_best, s_best, left_mean, right_mean, ll_score, &X, &UTy, &C, &S, &U, &noderange, delta);
     }
}
