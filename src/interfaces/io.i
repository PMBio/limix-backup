%shared_ptr(limix::CRMemDataFrame< MatrixXd >)
%shared_ptr(limix::CRWMemDataFrame< MatrixXd >)
%shared_ptr(limix::ARDataFrame< MatrixXd >)
%shared_ptr(limix::AGenotypeContainer);
%shared_ptr(limix::CTextfileGenotypeContainer);
%shared_ptr(limix::CMemGenotypeContainer);
%shared_ptr(limix::CGenotypeBlock);
%shared_ptr(limix::CHeaderMap);

//vector template definitions
namespace std {
	%template(StringStringVecMap) map<string,stringVec>;
};
