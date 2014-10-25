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

	void best_split_full_model(int* int_out,
		                       const MatrixXd& X,
		                       const MatrixXd& UTy,
		                       const MatrixXd& C,
		                       const MatrixXd& S,
		                       const MatrixXd& U,
		                       const VectorXi& noderange,
		                       mfloat_t delta){
   /* mfloat_t s_best = 0.0;*/
	//mfloat_t ll_score = 0.0;
	//mfloat_t left_mean = 0.0;
	//mfloat_t right_mean = 0.0;
    (*int_out) = 0;
    return;
    
       //C_best_split_full_model(out, &s_best, &left_mean, &right_mean, &ll_score, &X, &UTy, &C, &S, &U, &noderange, delta);
   }
   void argOutSwigTest2(int* int_out1, int* int_out2,mint_t in1,mint_t in2)
   {
   	(*int_out1) = in1;
   	(*int_out2) = in1+in2;
   }

   void argOutSwigTest3(float* int_out1, float* int_out2,mint_t in1,mint_t in2)
   {
   	(*int_out1) = in1;
   	(*int_out2) = in1+in2;
   }
}
