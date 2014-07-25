// Copyright(c) 2014, The LIMIX developers(Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

%shared_ptr(bost::enable_shared_from_this<CGPbase>)
%shared_ptr(std::map<std::string,MatrixXd>)
%shared_ptr(limix::CCovarianceFunctionCache)
%shared_ptr(limix::CCovarianceFunctionCacheOld)
%shared_ptr(limix::CGPCholCache)
%shared_ptr(limix::CGPHyperParams)
%shared_ptr(limix::CGPbase)
%shared_ptr(limix::CGPvarDecomp)
%shared_ptr(limix::CGPkronecker)
%shared_ptr(limix::CGPkronSum)
%shared_ptr(limix::CGPSum)
%shared_ptr(limix::CGPopt)
%shared_ptr(limix::CGPKroneckerCache)
%shared_ptr(limix::CGPkronSumCache)
%shared_ptr(limix::CGPSumCache)

//vector template definitions
namespace std {
   %template(MatrixXdVec) vector<MatrixXd>;
   %template(StringVec)   vector<string>;
   %template(StringMatrixMap) map<string,MatrixXd>;
};
