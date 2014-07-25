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

%shared_ptr(limix::CRMemDataFrame< MatrixXd >)
%shared_ptr(limix::CRWMemDataFrame< MatrixXd >)
%shared_ptr(limix::ARDataFrame< MatrixXd >)
%shared_ptr(limix::AGenotypeContainer);
%shared_ptr(limix::CTextfileGenotypeContainer);
%shared_ptr(limix::CMemGenotypeContainer);
%shared_ptr(limix::CGenotypeBlock);
%shared_ptr(limix::CHeaderMap);

//vector template definitions
/*
namespace std {
	%template(StringStringVecMap) map<string,stringVec>;
};
*/
