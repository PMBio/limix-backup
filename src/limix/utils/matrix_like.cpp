#include "matrix_like.h"
#include <stdlib.h>


namespace limix{


/***** AMatrixLike ******/
AMatrixLike::AMatrixLike()
{

}


AMatrixLike::~AMatrixLike()
{

}

void AMatrixLike::agetCholK(MatrixXdChol* out) throw (CGPMixException)
		{
		(*out) = MatrixXdChol(*this->getMatrix());
		}

void AMatrixLike::agetEigHK(MatrixXdEIgenSolver* out) throw(CGPMixException)
		{
		(*out) = MatrixXdEIgenSolver(*this->getMatrix());
		}

void AMatrixLike::agetMatrix(MatrixXd* out)
{
	*out = (*this->getMatrix());
}

/***** AMatrixLike ******/

AKronMatrixLike::AKronMatrixLike()
{
}

AKronMatrixLike::~AKronMatrixLike()
{
}


/***** CMATRIX ****/


CMatrix::CMatrix(PMatrixXd in)
{
	this->M = in;
}
CMatrix::CMatrix(const MatrixXd& in)
{
	M = PMatrixXd(new MatrixXd());
	(*M) = in;
}

PMatrixXd CMatrix::getMatrix()
{
	return this->M;
}



} // end:limix

