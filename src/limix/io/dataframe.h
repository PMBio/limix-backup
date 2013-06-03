/*
 * dataframe.h
 *
 *  Created on: May 16, 2013
 *      Author: stegle
 */

#ifndef DATAFRAME_H_
#define DATAFRAME_H_

#include "limix/types.h"

#include <string>
#include <map>
#include <vector>
#include <iostream>


namespace limix {

/*
 * Abstract class representing a data frame with row and columns covariance
 */

template <class MatrixType>
class ADataFrame {
public:

	ADataFrame()
	{};
	virtual ~ADataFrame()
	{};

	virtual void agetRowHeader(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PVectorXs getRowHeader() const throw(CGPMixException) = 0;

	virtual void agetColHeader(VectorXs* out) const throw(CGPMixException) = 0;
	virtual PVectorXs getColHeader() const throw(CGPMixException) = 0;

	virtual void agetMatrix(MatrixType* out) const throw (CGPMixException) = 0;
	virtual sptr<MatrixType> getMatrix() const throw (CGPMixException) =0;
};

template <class MatrixType>
class ARWDataFrame {
public:
	ARWDataFrame()
	{};
	virtual ~ARWDataFrame()
	{};

	virtual void setRowHeader(const VectorXs& in) throw (CGPMixException) = 0;
	virtual void setRowHeader(PVectorXs in) throw (CGPMixException) = 0;

	virtual void setColHeader(const VectorXs& in) throw (CGPMixException) = 0;
	virtual void setColHeader(PVectorXs in) throw (CGPMixException) = 0;

	virtual void setMatrix(const MatrixType& in) throw (CGPMixException) = 0;
	virtual void setMatrix(sptr<MatrixType> in) throw (CGPMixException) = 0;
};


//implementation for in-memory handling READ ONLY
template <class MatrixType>
class CMemDataFrame : public ADataFrame<MatrixType>
{
protected:
	sptr<MatrixType> M;
	PVectorXs rowHeader,colHeader;

	virtual void resizeMatrices(muint_t num_samples=-1,muint_t num_snps=-1);

public:
	CMemDataFrame()
	{
		M = sptr<MatrixType>(new MatrixType());
		rowHeader = PVectorXs(new VectorXs());
		colHeader = PVectorXs(new VectorXs());
	};
	CMemDataFrame(const CMemDataFrame<MatrixType>& copy)
	{
		this->M = copy.getMatrix();
		this->rowHeader = copy.getRowHeader();
		this->colHeader = copy.getColHeader();
	};


	CMemDataFrame(sptr<MatrixType> M,PVectorXs rowHeader,PVectorXs colHeader)
	{
	this->M = M;
	this->colHeader = colHeader;
	this->rowHeader = rowHeader;
	};

	virtual ~CMemDataFrame()
	{};

	virtual void agetRowHeader(VectorXs* out) const throw(CGPMixException)
	{
		(*out) = *rowHeader;
	}

	virtual PVectorXs getRowHeader() const throw(CGPMixException)
	{
		return rowHeader;
	}

	virtual void agetColHeader(VectorXs* out) const throw(CGPMixException)
	{
		(*out) = *colHeader;
	}

	virtual PVectorXs getColHeader() const throw(CGPMixException)
	{
			return rowHeader;
	}

	virtual void agetMatrix(MatrixType* out) const throw (CGPMixException)
	{
		(*out) = *M;
	}
	virtual sptr<MatrixType> getMatrix() const throw (CGPMixException)
	{
		return M;
	}
};

//implementation for in-memory handling READ ONLY
template <class MatrixType> class CMemRWDataFrame : public CMemDataFrame<MatrixType>, public ARWDataFrame<MatrixType>
{
public:
	CMemRWDataFrame()
	{};
	CMemRWDataFrame(const CMemDataFrame<MatrixType>& copy) : CMemDataFrame<MatrixType>(copy)
	{};


	virtual ~CMemRWDataFrame()
	{};

	virtual void setRowHeader(const VectorXs& in) throw (CGPMixException)
		{
		(*this->rowHeader) = in;
		}
	virtual void setRowHeader(PVectorXs in) throw (CGPMixException)
		{
		this->rowHeader = in;
		}


	virtual void setColHeader(const VectorXs& in) throw (CGPMixException)
		{
		(*this->colHeader) = in;
		}
	virtual void setColHeader(PVectorXs in) throw (CGPMixException)
		{
		this->colHeader = in;
		}

	virtual void setMatrix(const MatrixType& in) throw (CGPMixException)
		{
		(*this->M) = in;
		}
	virtual void setMatrix(sptr<MatrixType> in) throw (CGPMixException)
		{
		this->M = in;
		}

};


#if (defined(SWIG) && !defined(SWIG_FILE_WITH_INIT))
%template(CMemDataFrameXd) CMemDataFrame<MatrixXd>;
%template(CMemRWDataFrameXd) CMemRWDataFrame<MatrixXd>;
%template(ADataFrameXd) ADataFrame< MatrixXd >;
%template(ARWDataFrameXd) ARWDataFrame< MatrixXd >;
#endif


}
		//end: namespace limix

template<class MatrixType>
inline void limix::CMemDataFrame<MatrixType>::resizeMatrices(
		muint_t num_rows, muint_t num_columns) {
this->M->resize(num_rows,num_columns);
this->rowHeader->resize(num_rows);
this->colHeader->resize(num_columns);
}

#endif /* DATAFRAME_H_ */
