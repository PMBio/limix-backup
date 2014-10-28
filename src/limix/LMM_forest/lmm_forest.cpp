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
		                       mfloat_t delta){
    (*m_best) = 0;
    (*s_best) = 1.0;
    (*left_mean) = 2.0;
    (*right_mean) = 4.0;
    (*ll_score) = 5.0;

    mint_t m_best_ = 0;
    mfloat_t s_best_ = 0.0;
    mfloat_t  right_mean_ = 0.0;
    mfloat_t  left_mean_ = 0.0;
    mfloat_t  ll_score_ = 0.0;
   
    
   C_best_split_full_model(&m_best_, &s_best_, &left_mean_, &right_mean_, &ll_score_, &X, &UTy, &C, &S, &U, &noderange, delta);
   std::cout << m_best;
   return;
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
