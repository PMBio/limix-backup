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

namespace limix {

}

//raw include
//apply derivatices:
%apply float *OUTPUT {float* m_best, float* s_best, float* left_mean, float* right_mean, float* ll_score};
%apply int *OUTPUT { int* int_out1, int* int_out2};
%apply float *OUTPUT { float* int_out1, float* int_out2};

%include "limix/LMM_forest/lmm_forest.h"
%include "limix/types.h"
