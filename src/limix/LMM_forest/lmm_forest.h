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
	void best_split_full_model(float* m_best,
                               float* s_best,
                               float* left_mean,
                               float* right_mean,
                               float* ll_score,
		                       const MatrixXd& X,
		                       const MatrixXd& UTy,
		                       const MatrixXd& C,
		                       const MatrixXd& S,
		                       const MatrixXd& U,
		                       const VectorXi& noderange,
		                       mfloat_t delta);


    void predict_lmm_forest(MatrixXd* response,
                 const VectorXi& tree_nodes,
                 const VectorXi& left_children,
                 const VectorXi& right_children,
                 const VectorXi& best_predictor,
                 const MatrixXd& mean,
                 const MatrixXd& splitting_value,
                 const MatrixXd& X,
                 mfloat_t depth);
                
	void argOutSwigTest2(int* int_out1, int* int_out2,mint_t in1,mint_t in2);
	void argOutSwigTest3(float* int_out1, float* int_out2,mint_t in1,mint_t in2);
	void argOutSwigTest4(int* int_out1, int* int_out2,const MatrixXd& m);
}
#endif
