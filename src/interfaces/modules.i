// Copyright(c) 2014, The LIMIX developers (Christoph Lippert, Paolo Francesco Casale, Oliver Stegle)
// All rights reserved.
//
// LIMIX is provided under a 2-clause BSD license.
// See license.txt for the complete license.

%shared_ptr(limix::CVqtl)
%shared_ptr(limix::CMultiTraitVQTL)
%shared_ptr(limix::AVarianceTerm)
%shared_ptr(limix::CSingleTraitTerm)
%shared_ptr(limix::CMultiTraitTerm)
%shared_ptr(limix::CVarianceDecomposition)

//vector template definitions
namespace std {
   %template(VectorXiVec) vector<VectorXi>;
};
